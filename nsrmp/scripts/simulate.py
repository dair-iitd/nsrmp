import os
import argparse
import torch
import pybullet as p

import os
import json
import shutil

from data_generation import construct, get_scenes_json, get_instructions_json
from datasets.roboclevr.definition import build_nsrm_dataset
from model.configs import configs as model_configs 
from model.model_new import Model
from helpers.mytorch.cuda.copy import async_copy_to
from data_generation.panda.construct.base import ConstructBase
from scripts.pixel_to_world import load_model
from helpers.mytorch.vision.ops.boxes import box_convert
from helpers.utils.type_conversion import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--record', type=str2bool, default = False)
parser.add_argument('--video_filename', type=str, default = 'nsrm_recorded.mp4')
parser.add_argument('--width', default=1024, help='Width of GUI Window')
parser.add_argument('--height', default=768, help='Height of GUI Window')

parser.add_argument('--model_path', default="./model_saves/model_final_single_step_relational.pth")

parser.add_argument('--example_path', default='./examples/00001')
parser.add_argument('--predicted', type=str2bool, default = True)
parser.add_argument('--adjust_horizontal', default = True)

parser.add_argument('--datadir', type = str, default = '../data/')
parser.add_argument('--vocab_json', default='./nsrmp/vocab_new.json')
parser.add_argument('--training_target', default = 'splitter')
parser.add_argument('--use_cuda', type=bool, default = True)
parser.add_argument('--instruction_transform', type = str, default = 'basic')
parser.add_argument('--batch_size', default = True)
parser.add_argument('--use_gt_grasp', default = False)
args = parser.parse_args()

if os.path.exists(os.path.join(args.datadir, 'sample')):
	shutil.rmtree(os.path.join(args.datadir, 'sample'))
os.makedirs(os.path.join(args.datadir, 'sample'))

new_example_path = os.path.join(args.datadir, 'sample', os.path.basename(os.path.normpath(args.example_path)))
shutil.copytree(args.example_path, new_example_path)

config = {
	'demo_json_path': os.path.join(new_example_path, 'demo.json')
}

movements = []

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

	pixel2world_model = load_model()
	if args.use_cuda:
		pixel2world_model.cuda()

	for (bbox_orig, move_obj_mask, bbox) in outputs['movements'][0]:
		nonzeros = torch.nonzero(move_obj_mask)
		assert len(nonzeros) == 1, "Only one object can be moved!"
		move_obj_idx = nonzeros[0][0]

		bbox = box_convert(bbox, model_configs.data.bbox_mode, 'xywh')

		bbox_orig = box_convert(bbox_orig, model_configs.data.bbox_mode, 'xywh')

		target_pos = pixel2world_model(bbox)
		initial_pos = pixel2world_model(bbox_orig)
		movements.append((move_obj_idx, list(target_pos), list(initial_pos)))

else:
	# Ideal movement
	with open(config['demo_json_path'], 'r') as f:
		demo_data = json.load(f)
		for i,prog in enumerate(demo_data['grounded_program']):
			move_obj_idx = prog[1]
			target_pos = demo_data['object_positions'][i+1][move_obj_idx] # For ith program, check pos at i+1th frame
			movements.append((move_obj_idx, target_pos, None))

timeStep = 1/240.0
if args.record:
	construct.init_bulletclient(timeStep, args.width, args.height, args.video_filename)
else:
	construct.init_bulletclient(timeStep)


constructBase = ConstructBase(p, [0,0,0], config, args.height, args.width, None, set_hide_panda_body=False)

for move_obj_idx, target_pos, initial_pos in movements:
	if args.use_gt_grasp:
		initial_pos = None
	constructBase.move_object(move_obj_idx, target_pos, use_panda=True, initial_pos=initial_pos, adjust_horizontal=args.adjust_horizontal)

