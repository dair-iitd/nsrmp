#####################
### Test Baseline ###
#####################

# Standard imports
import os
import sys
import math
import time
import argparse
from tqdm import tqdm

# PyTorch related imports
import torch

# Helper imports
from helpers.logging import get_logger
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.mytorch.train.freeze import mark_freezed
from datasets.roboclevr.definition import build_nsrm_dataset

# Baseline related imports
from baseline.baselineModel import BaselineModel, BaselineModelSplitter
from baseline.configs import configs as model_configs 

args = argparse.ArgumentParser()

# [REQUIRED] Arguments
args.add_argument('--datadir', type=str, default=None)
args.add_argument('--type', type=str, default=None)
args.add_argument('--load_model', type=str, default=None)

# [OPTIONAL] Arguments
args.add_argument('--use_cuda', type=bool, default=True)
args.add_argument('--num_steps', type=int, default=[0, 2], nargs="+")
args.add_argument('--num_objects', type=int, default=[0, 5], nargs="+")
args.add_argument('--language_complexity', type=str, default=None)
args.add_argument('--remove_relational', type=bool, default=False)
args.add_argument('--only_relational', type=bool, default=False)

args = args.parse_args()
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d---%H-%M-%S'))

# [IMMUTABLE] Arguments
args.batch_size = 1
args.instruction_transform = 'off'

assert args.datadir is not None, "--datadir field is empty"
test_data = {
    'dir': os.path.join(args.datadir, 'test'),
    'scenes': os.path.join(args.datadir, 'scenes-test.json'),
    'instructions': os.path.join(args.datadir, 'instructions-test.json'),
    'vocab': os.path.join(args.datadir, 'vocab.json'),
}

logger = get_logger(__file__)

logger.critical("Building the test dataset")
test_dataset = build_nsrm_dataset(args, model_configs, test_data['dir'], test_data['scenes'], test_data['instructions'], test_data['vocab'])

assert args.type is not None, "--type field is empty"
logger.critical("Building the model")
if args.type == 'single':
    model = BaselineModel(test_dataset.vocab, model_configs)
elif args.type == 'multi':
    model = BaselineModelSplitter(test_dataset.vocab, model_configs)
else:
    sys.exit("Invalid baseline type.")

if args.use_cuda:
    model.cuda()

def testing_loop(dataset, name="test"):
    dataloader = dataset.make_dataloader(args.batch_size)
    argO1, argO2, iou2D_move, iou2D_move_var, iou2D_total, i = 0, 0, 0, 0, 0, 0
    for _, batch in tqdm(enumerate(dataloader)):
        if args.use_cuda:
            batch = async_copy_to(batch,dev = 0, main_stream=None)
        with torch.no_grad():
            temp1, temp2, temp3, temp4, temp5 = model.inference(batch)
        argO1 += temp1
        argO2 += temp2
        iou2D_move += temp3
        iou2D_move_var += temp4
        iou2D_total += temp5
        i += 1
    print(f"Performance on {name} set")
    print("Mean Arg O1: ", 100 * (argO1 / (i * args.batch_size)))
    print("Mean Arg O2: ", 100 * (argO2 / (i * args.batch_size)))
    print("Mean IoU 2D (Move): ", 100 * (iou2D_move / (i * args.batch_size)).item())
    print("Std IoU 2D (Move)", ((torch.sqrt(iou2D_move_var / (i * args.batch_size) - (iou2D_move / (i * args.batch_size)) ** 2)) / math.sqrt(i * args.batch_size)).item())
    print("Mean IoU 2D (Total): ", 100 * (iou2D_total / (i * args.batch_size)).item())

def test_unit(filter):
    this_test_dataset = test_dataset
    this_test_dataset = this_test_dataset.filter_step(filter[0])
    this_test_dataset = this_test_dataset.filter_scene_size(filter[1])
    if filter[2] != None:
        this_test_dataset = this_test_dataset.filter_language_complexity(filter[2])
    if filter[3]:
        this_test_dataset = this_test_dataset.remove_relational()
    if filter[4]:
        this_test_dataset = this_test_dataset.filter_relational()
    mark_freezed(model)
    testing_loop(this_test_dataset)

if __name__ == '__main__':
    assert args.load_model is not None, "--load_model field is empty"
    logger.critical('Trying to load model from {}'.format(args.load_model))
    try:
        saved_model_state_dict = torch.load(args.load_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    except:
        sys.exit("The keys being loaded are incompatible with the current architecture of the model. The model architecture might have changed significantly from the last checkpoint.")
    logger.critical('Loaded the model succesfully!\n')
    test_unit([args.num_steps, args.num_objects, args.language_complexity, args.remove_relational, args.only_relational])
