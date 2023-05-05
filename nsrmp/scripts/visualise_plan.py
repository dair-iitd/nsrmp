'''
Currently only works if any given object is moved atmost once (i.e. an object moved once is not moved again in further steps)
To make it work for cases not included above, we need to save intermediate scenes in dataset 

Note: 
pip uninstall opencv-python
pip install --no-binary opencv-python opencv-python
'''

import os
import argparse
import torch
import cv2
import numpy as np
import os
import json
import shutil

from data_generation import get_scenes_json, get_instructions_json
from datasets.roboclevr.definition import build_nsrm_dataset
from model.configs import configs as model_configs 
from model.model_new import Model
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.mytorch.vision.ops.boxes import box_convert
from helpers.utils.type_conversion import str2bool

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--record', action="store_true", help='Dataset Directory From Root')
parser.add_argument('--width', default=1024, help='Width of GUI Window')
parser.add_argument('--height', default=768, help='Height of GUI Window')
parser.add_argument('--root_dir', default=os.getcwd(), metavar='DIR', help='Root Directory')

parser.add_argument('--model_path', default="./model_saves/model_final_single_step_relational.pth")
parser.add_argument('--example_path', default='./examples/00001')
parser.add_argument('--datadir', type = str, default = '../data/')
parser.add_argument('--vocab_json', default='./nsrmp/vocab_new.json')
parser.add_argument('--training_target', default = 'splitter')
parser.add_argument('--use_cuda', type=bool, default = True)
parser.add_argument('--instruction_transform', type = str, default = 'basic')
parser.add_argument('--predicted', type=str2bool, default = True)
parser.add_argument('--batch_size', default = True)
args = parser.parse_args()


shutil.rmtree(os.path.join(args.datadir, 'sample'))
os.makedirs(os.path.join(args.datadir, 'sample'))

new_example_path = os.path.join(args.datadir, 'sample', os.path.basename(os.path.normpath(args.example_path)))
shutil.copytree(args.example_path, new_example_path)

demo_json_path = os.path.join(new_example_path, 'demo.json')
initial_scene_path = os.path.join(new_example_path, 'S00', 'rgba.png')
scenes_json_path = os.path.join(args.datadir, 'scenes-sample.json')

with open(scenes_json_path, 'r') as f:
	scenes_data = json.load(f)

with open(demo_json_path, 'r') as f:
	demo_data = json.load(f)

img = cv2.imread(initial_scene_path)

def plot_bbox(bbox, step, img, is_initial):
	if is_initial:
		color = (255, 0, 0)
	else:
		color = (0, 255, 0)
	y,x,h,w,_ = bbox
	img = cv2.rectangle(img, (int(x*args.width),int(y*args.height)), (int((x+w)*args.width),int((y+h)*args.height)), color, 2)
	img = cv2.putText(img, str(step+1), (int(x*args.width),int(y*args.height)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
	return img 

if args.predicted:
	get_scenes_json.main(os.path.join(args.datadir, 'sample'), 'sample', 'scenes-sample.json', out_dir=args.datadir)
	get_instructions_json.main(os.path.join(args.datadir, 'sample'), 'sample', 'instructions-sample.json', args.datadir)

	sample_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'sample'), os.path.join(args.datadir, 'scenes-sample.json'), os.path.join(args.datadir, 'instructions-sample.json'), args.vocab_json)

	model = Model(sample_dataset.vocab, model_configs ,training_target = args.training_target)
	model.prepare_model('usual')
	
	if args.use_cuda:
		model.cuda()

	from helpers.mytorch.base.serialization import load_state_dict
	load_state_dict(model, args.model_path, partial = True, modules = ['parser','resnet','visual','action_sim','concept_embeddings'])
	load_state_dict(model,'./model_saves/splitter_relational_2.pth', partial = True, modules = ['multi_step_builder'])
	model.eval()

	kwargs = dict(unique_mode = 'argmax', gumbel_tau = 0.00001)

	dataloader = sample_dataset.make_dataloader(1, shuffle = False, sampler = None, drop_last = True)
	for batch in dataloader:
		if args.use_cuda:
			batch = async_copy_to(batch,dev=0, main_stream=None)
		outputs = model(batch, **kwargs)
	for step, (bbox_i, move_obj_mask, bbox) in enumerate(outputs['movements'][0]):
		nonzeros = torch.nonzero(move_obj_mask)
		assert len(nonzeros) == 1, "Only one object can be moved!"
		move_obj_idx = nonzeros[0][0]

		bbox = box_convert(bbox, model_configs.data.bbox_mode, 'xywh')

		initial_bbox = box_convert(bbox_i, model_configs.data.bbox_mode, 'xywh')

		# initial_bbox = scenes_data['scenes'][0][0]['objects'][move_obj_idx]['bbox']
        
		img = plot_bbox(initial_bbox, step, img, True)
		img = plot_bbox(bbox, step, img, False)

	cv2.imwrite('plan-predicted.png', img)  

else:
	# Ideal movement
	for i,prog in enumerate(demo_data['grounded_program']):
		move_obj_idx = prog[1]

		initial_bbox = scenes_data['scenes'][0][0]['objects'][move_obj_idx]['bbox']

		final_bbox = scenes_data['scenes'][0][1]['objects'][move_obj_idx]['bbox']

		img = plot_bbox(initial_bbox, i, img, True)
		img = plot_bbox(final_bbox, i, img, False)

	cv2.imwrite('plan-gt.png', img)  
