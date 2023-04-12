# basic imports
import os
import sys
import time
import argparse
from tqdm import tqdm
from pathlib import Path

# torch related imports
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
# from torch.utils.tensorboard import SummaryWriter

# helpers
from helpers.logging import get_logger
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.mytorch.train.freeze import mark_unfreezed, mark_freezed

# packages within this repo
from baseline.baselineModel import BaselineModelOld, BaselineModelMultiStep
from baseline.executor import BaselineModelExecutor
# from baseline_.baselineModel import BaselineModel
from helpers.utils.container import DOView
from baseline.configs import configs as model_configs 
from datasets.roboclevr.definition import build_nsrm_dataset
import json
import shutil
from data_generation import construct, get_scenes_json, get_instructions_json
from scripts.pixel_to_world import load_model
from data_generation.panda.construct.base import ConstructBase
from helpers.mytorch.vision.ops.boxes import box_convert
import pybullet as p
from helpers.utils.type_conversion import str2bool

args = argparse.ArgumentParser()

args.add_argument('--example_path', default='/home/vishal/projects/nsrmp/data_new/test/00823')
args.add_argument('--load_model_from_file',type = str, default='/home/arnavtuli/model_multi_step_rel_iou.pth')

args.add_argument('--datadir', type = str, default = '../data/')

args.add_argument('--adjust_horizontal', default = True)
args.add_argument('--use_gt_grasp', default = False)

args.add_argument('--record', type=str2bool, default = True)
args.add_argument('--video_filename', type=str, default = '/home/vishal/projects/nsrmp/nsrmp/recorded.mp4')
args.add_argument('--width', default=1024, help='Width of GUI Window')
args.add_argument('--height', default=768, help='Height of GUI Window')

#Added by namas for dataloader
args.add_argument('--dataset', type=str, default='roboclevr')
args.add_argument('--vocab_json', default='/home/namas/Desktop/nsrmp/data/vocab.json')
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--num_epochs', type=int, default=200)
args.add_argument('--use_cuda', type = bool, default=True)
args.add_argument('--eval_interval',type = int, default=10)
args.add_argument('--model_save_interval', type = int, default = 100)
args.add_argument('--model_save_dir',type = str, default="/home/arnavtuli/")
args.add_argument('--save_model_to_file', type = str, default=None)
args.add_argument('--wandb',type = bool, default=False)
args.add_argument('--type_baseline',type = str, default='new')
args.add_argument('--decompose', type=bool, default=False)

#########################################################
###############       UNUSED ARGS       #################
#########################################################
args.add_argument('--save_vocab', type=bool, default=False)
args.add_argument('--training_target',type =str, default = 'all')
args.add_argument('--instruction_transform', type = str, default = 'basic')
args.add_argument('--use_condensed_representation',type=bool,default=False)
args.add_argument('--create_condensed_representation',type=bool,default=False)
args.add_argument('--model',default="model")
args.add_argument('--validation_logger_freq',default=10)
args.add_argument('--plot_iou',default=False)
args.add_argument('--save_object_embeddings_to_file',default=None)
args.add_argument('--save_concept_embeddings_to_file',default=None)
args.add_argument('--load_object_embeddings_from_file',default=None)
args.add_argument('--load_concept_embeddings_from_file',default=None)
args.add_argument('--freeze_concept_embeddings',default=False)
args.add_argument('--freeze_object_embeddings',default=False)
args.add_argument('--test_pkl', type=str, default=None)
args.add_argument('--parser', type=str)
args.add_argument('--use_grounded_programs', default=False)
args.add_argument('--plot_bboxes_mode', default=1) # 0-None, 1-Only one example, 2-all
args.add_argument('--grounded_programs_supervision',default=False)
args.add_argument('--use_symbolic_programs',default=False)
args.add_argument('--exact_expected_iou',default=False)
#########################################################

args = args.parse_args()

if os.path.exists(os.path.join(args.datadir, 'sample')):
	shutil.rmtree(os.path.join(args.datadir, 'sample'))
os.makedirs(os.path.join(args.datadir, 'sample'))

new_example_path = os.path.join(args.datadir, 'sample', os.path.basename(os.path.normpath(args.example_path)))
shutil.copytree(args.example_path, new_example_path)

config = {
	'demo_json_path': os.path.join(new_example_path, 'demo.json')
}

get_scenes_json.main(os.path.join(args.datadir, 'sample'), 'sample', 'scenes-sample.json', out_dir=args.datadir)
get_instructions_json.main(os.path.join(args.datadir, 'sample'), 'sample', 'instructions-sample.json', args.datadir)

logger = get_logger(__file__)
logger.critical("Building the dataset")
sample_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, "sample"), os.path.join(args.datadir, 'scenes-sample.json'), os.path.join(args.datadir, 'instructions-sample.json'), args.vocab_json)

logger.critical("Building the Model")
model = BaselineModelExecutor(sample_dataset.vocab, model_configs)

if args.use_cuda:
	model.cuda()

pixel2world_model = load_model()
if args.use_cuda:
	pixel2world_model.cuda()

def get_model_outputs():
	torch.autograd.set_detect_anomaly(True)
	if args.load_model_from_file is not None : 
		logger.critical('Trying to load model from {}'.format(args.load_model_from_file))
		try:
			saved_model_state_dict = torch.load(os.path.join(args.model_save_dir,args.load_model_from_file))
			model_dict = model.state_dict()
			pretrained_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_dict}
			model_dict.update(pretrained_dict)
			model.load_state_dict(model_dict)
		except:
			sys.exit("The keys being loaded is incompatible with the current architechture of the model. The model architechture might changed significantly from the last checkpoint. ")
		logger.critical('Loaded the model succesfully!\n')

	mark_freezed(model)

	sample_dataloader = sample_dataset.make_dataloader(args.batch_size, sampler = None, drop_last = True)
	pbar = tqdm(enumerate(sample_dataloader))
	for iteration, batch in pbar:
		if args.use_cuda:
			batch = async_copy_to(batch,dev = 0, main_stream=None)
		
		outputs = model(batch)
	return outputs

movements = get_model_outputs()

timeStep = 1/240.0
if args.record:
	video_filename = '/home/vishal/Desktop/videos/'+os.path.basename(os.path.normpath(args.example_path))+'-baseline.mp4'
	assert not os.path.exists(video_filename)
	construct.init_bulletclient(timeStep, args.width, args.height, video_filename)
else:
	construct.init_bulletclient(timeStep)


constructBase = ConstructBase(p, [0,0,0], config, args.height, args.width, None, set_hide_panda_body=False)

for (bbox, move_obj_idx, bbox_orig) in movements:
		bbox = box_convert(bbox.view(-1), model_configs.data.bbox_mode, 'xywh')
		bbox_orig = box_convert(bbox_orig, model_configs.data.bbox_mode, 'xywh')

		target_pos = list(pixel2world_model(bbox))
		initial_pos = list(pixel2world_model(bbox_orig))
		
		# breakpoint()
		if args.use_gt_grasp:
			initial_pos = None
		constructBase.move_object(move_obj_idx, target_pos, use_panda=True, initial_pos=initial_pos, adjust_horizontal=args.adjust_horizontal)
