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

args.add_argument('--dataset', type=str, default='roboclevr')
args.add_argument('--datadir', type = str, default = '/home/himanshu/Desktop')
args.add_argument('--train_scenes_json' ,type=str, default ='scenes-train.json')
args.add_argument('--train_instructions_json', type=str, default = 'instructions-train.json')
args.add_argument('--test_scenes_json', type = str, default= 'scenes-test.json')
args.add_argument('--test_instructions_json', type = str, default= 'instructions-test.json')
args.add_argument('--vocab_json', default=None)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--use_cuda', type=bool, default=False)
args.add_argument('--model_save_dir', type=str, default="/home/arnavtuli/nsrmp/nsrmp/model_saves")
args.add_argument('--load_model_from_file', type=str, default=None)
args.add_argument('--load_splitter', type=bool, default=False)
args.add_argument('--type_baseline', type=str, default='single')
args.add_argument('--instruction_transform', type=str, default='off')
args.add_argument('--data_assoc', type=bool, default=True)
args.add_argument('--use_iou_loss', type=bool, default=True)
args.add_argument('--use_train', type=bool, default=True)
args.add_argument('--num_steps', type=int, default=[0, 2], nargs="+")
args.add_argument('--num_objects', type=int, default=[0, 5], nargs="+")
args.add_argument('--language_complexity', type=str, default=None)
args.add_argument('--remove_relational', type=bool, default=False)
args.add_argument('--only_relational', type=bool, default=False)


args = args.parse_args()
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d---%H-%M-%S'))

args.train_scenes_json = os.path.join(args.datadir,args.train_scenes_json)
args.train_instructions_json = os.path.join(args.datadir,args.train_instructions_json)
args.test_scenes_json = os.path.join(args.datadir,args.test_scenes_json)
args.test_instructions_json = os.path.join(args.datadir,args.test_instructions_json)

model_configs.model.use_iou_loss = args.use_iou_loss
model_configs.model.data_assoc = args.data_assoc

logger = get_logger(__file__)

logger.critical("Building the train and test dataset")
train_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, "train"), args.train_scenes_json, args.train_instructions_json, args.vocab_json)
test_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, "test"), args.test_scenes_json, args.test_instructions_json, args.vocab_json)

logger.critical("Building the Model")
if args.type_baseline == 'single':
    model = BaselineModel(test_dataset.vocab, model_configs)
elif args.type_baseline == 'multi':
    model = BaselineModelSplitter(test_dataset.vocab, model_configs)
else:
    sys.exit("Invalid baseline type.")

if args.use_cuda:
    model.cuda()

def testing_loop(dataset, name="Test"):
    dataloader = dataset.make_dataloader(args.batch_size, drop_last=True)
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
    if args.use_train:
        this_train_dataset = train_dataset
    this_test_dataset = test_dataset
    # Filter dataset
    if filter[0] != None:
        if args.use_train:
            this_train_dataset = this_train_dataset.filter_step(filter[0])
        this_test_dataset = this_test_dataset.filter_step(filter[0])
    if filter[1] != None:
        if args.use_train:
            this_train_dataset = this_train_dataset.filter_scene_size(filter[1])
        this_test_dataset = this_test_dataset.filter_scene_size(filter[1])
    if filter[2] != None:
        if args.use_train:
            this_train_dataset = this_train_dataset.filter_language_complexity(filter[2])
        this_test_dataset = this_test_dataset.filter_language_complexity(filter[2])
    if filter[3]:
        if args.use_train:
            this_train_dataset = this_train_dataset.remove_relational()
        this_test_dataset = this_test_dataset.remove_relational()
    if filter[4]:
        if args.use_train:
            this_train_dataset = this_train_dataset.only_relational()
        this_test_dataset = this_test_dataset.only_relational()
    # Set model for testing
    mark_freezed(model)
    testing_loop(this_test_dataset, "Test")
    if args.use_train:
        testing_loop(this_train_dataset, "Train")

if __name__ == '__main__':
    # Load previously trained model, if any
    if args.load_model_from_file is not None : 
        logger.critical('Trying to load model from {}'.format(args.load_model_from_file))
        try:
            saved_model_state_dict = torch.load(os.path.join(args.model_save_dir,args.load_model_from_file))
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except:
            sys.exit("The keys being loaded is incompatible with the current architechture of the model. The model architechture might have changed significantly from the last checkpoint. ")
        logger.critical('Loaded the model succesfully!\n')

    # Load splitter separately, if needed
    if args.load_splitter and args.type_baseline == 'multi':
        saved_splitter_embed = torch.load(os.path.join(args.model_save_dir, "splitter_embed.pth"))
        model_embed_dict = model.embed.state_dict()
        pretrained_dict = {k: v for k, v in saved_splitter_embed.items() if k in model_embed_dict}
        model_embed_dict.update(pretrained_dict)
        model.embed.load_state_dict(model_embed_dict)
        
        saved_splitter_lstm = torch.load(os.path.join(args.model_save_dir, "splitter_lstm.pth"))
        model_lstm_dict = model.lstm.state_dict()
        pretrained_dict = {k: v for k, v in saved_splitter_lstm.items() if k in model_lstm_dict}
        model_lstm_dict.update(pretrained_dict)
        model.lstm.load_state_dict(model_lstm_dict)
        
        saved_splitter_linear = torch.load(os.path.join(args.model_save_dir, "splitter_linear.pth"))
        model_linear_dict = model.linear.state_dict()
        pretrained_dict = {k: v for k, v in saved_splitter_linear.items() if k in model_linear_dict}
        model_linear_dict.update(pretrained_dict)
        model.linear.load_state_dict(model_linear_dict)
        
        print("Sentence splitter loaded successfully!")

    # Test model
    test_unit([args.num_steps, args.num_objects, args.language_complexity, args.remove_relational, args.only_relational])
