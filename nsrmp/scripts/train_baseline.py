######################
### Train Baseline ###
######################

# Standard imports
import os
import sys
import time
import argparse
from tqdm import tqdm
from pathlib import Path

# PyTorch related imports
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Helper imports
from helpers.logging import get_logger
from helpers.mytorch.cuda.copy import async_copy_to
from datasets.roboclevr.definition import build_nsrm_dataset
from helpers.mytorch.train.freeze import mark_unfreezed

# Baseline related imports
from baseline.baselineModel import BaselineModel, BaselineModelSplitter
from baseline.configs import configs as model_configs 

args = argparse.ArgumentParser()

# [REQUIRED] Arguments
args.add_argument('--datadir', type=str, default=None)
args.add_argument('--type', type=str, default=None)

# [OPTIONAL] Arguments
args.add_argument('--use_cuda', type=bool, default=True)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--num_epochs', type=int, default=200)
args.add_argument('--save_model', type=str, default=None)
args.add_argument('--load_model', type=str, default=None)
args.add_argument('--save_splitter', type=bool, default=False)
args.add_argument('--load_splitter', type=bool, default=False)
args.add_argument('--save_interval', type=int, default=5)
args.add_argument('--data_assoc', type=bool, default=True)
args.add_argument('--use_iou_loss', type=bool, default=False)
args.add_argument('--train_splitter', type=bool, default=False)
args.add_argument('--num_steps', type=int, default=[0, 2], nargs="+")
args.add_argument('--num_objects', type=int, default=[0, 5], nargs="+")
args.add_argument('--language_complexity', type=str, default=None)
args.add_argument('--remove_relational', type=bool, default=False)
args.add_argument('--only_relational', type=bool, default=False)
args.add_argument('--wandb', type=bool, default=False)

args = args.parse_args()
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d---%H-%M-%S'))

# [IMMUTABLE] Arguments
args.instruction_transform = 'off'

assert args.datadir is not None, "--datadir field is empty"
train_data = {
    'dir': os.path.join(args.datadir, 'train'),
    'scenes': os.path.join(args.datadir, 'scenes-train.json'),
    'instructions': os.path.join(args.datadir, 'instructions-train.json'),
    'vocab': os.path.join(args.datadir, 'vocab.json'),
}
val_data = {
    'dir': os.path.join(args.datadir, 'val'),
    'scenes': os.path.join(args.datadir, 'scenes-val.json'),
    'instructions': os.path.join(args.datadir, 'instructions-val.json'),
    'vocab': os.path.join(args.datadir, 'vocab.json'),
}

model_configs.model.train_splitter = args.train_splitter
model_configs.model.use_iou_loss = args.use_iou_loss
model_configs.model.data_assoc = args.data_assoc

logger = get_logger(__file__)

logger.critical("Building the training and validation datasets")
train_dataset = build_nsrm_dataset(args, model_configs, train_data['dir'], train_data['scenes'], train_data['instructions'], train_data['vocab'])
val_dataset = build_nsrm_dataset(args, model_configs, val_data['dir'], val_data['scenes'], val_data['instructions'], val_data['vocab'])

logger.critical("Building the model")
assert args.type is not None, "--type field is empty"
if args.type == 'single':
    model = BaselineModel(train_dataset.vocab, model_configs)
elif args.type == 'multi':
    model = BaselineModelSplitter(train_dataset.vocab, model_configs)
else:
    sys.exit("Invalid baseline type.")

if args.use_cuda:
    model.cuda()

if args.wandb:
    import wandb
    wandb.init(project="nsrm-baseline", entity="nsrmp")
    wandb.watch(model, log='all', log_freq=1)

def validation(val_dataset, name="validation"):
    val_dataloader = val_dataset.make_dataloader(args.batch_size)
    losses, ious, subject_attn, object_attn = [], [], [], []
    model.eval()
    for batch in tqdm(val_dataloader):
        if args.use_cuda:
            batch = async_copy_to(batch, dev=0, main_stream=None)
        with torch.no_grad():
            if args.type == 'single' or not model_configs.model.train_splitter:
                loss, loss_details, sub_attn, obj_attn = model(batch)
            else:
                loss, loss_details = model(batch)
        losses.append(loss.item())
        ious.append(sum(loss_details['ious']).item() / len(loss_details['ious']))
        if args.type == 'single' or not model_configs.model.train_splitter:
            subject_attn.append(sub_attn)
            object_attn.append(obj_attn)
    
    logger.critical(f"Mean loss: {sum(losses) / len(losses)}")
    logger.critical(f"IOU: min_iou = {min(ious)}, max_iou = {max(ious)}, mean_iou = {sum(ious) / len(ious)}")
    if args.wandb:
        if args.type == 'single' or not model_configs.model.train_splitter:
            wandb.log({
            f'{name} mean-loss': torch.tensor(losses).mean().item(),
            f'{name} mean-iou': torch.tensor(ious).mean().item(),
            f'{name} sub-mean-attn': torch.tensor(subject_attn).mean().item(),
            f'{name} obj-mean-attn': torch.tensor(object_attn).mean().item(),
            })
        else:
             wandb.log({
            f'{name} mean-loss': torch.tensor(losses).mean().item(),
            f'{name} mean-iou': torch.tensor(ious).mean().item(),
            })
    model.train()


def training_loop(train_dataset, val_dataset, num_epochs, optimizer, scheduler):
    for epoch in range(num_epochs):
        train_dataloader = train_dataset.make_dataloader(args.batch_size)
        losses, ious, subject_attn, object_attn = [],[], [], []
        pbar = tqdm(enumerate(train_dataloader))
        for iteration, batch in pbar:
            if args.use_cuda:
                batch = async_copy_to(batch, dev=0, main_stream=None)
            try:
                if args.type == 'single' or not model_configs.model.train_splitter:
                    loss, loss_details, sub_attn, obj_attn = model(batch)
                else:
                    loss, loss_details = model(batch)
            except Exception as _:
                logger.error(f'\nError in epoch {epoch} batch {iteration}', exc_info=True, stack_info=True)
                exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.type == 'single' or not model_configs.model.train_splitter:
                subject_attn.append(sub_attn)
                object_attn.append(obj_attn)
            ious.append(sum(loss_details['ious']).item() / len(loss_details['ious']))
            
            pbar.set_description(f'Epoch: {epoch}, Loss:{loss.item()}' )
            if iteration % 2 == 0:
                if args.wandb:
                    if args.type == 'single' or not model_configs.model.train_splitter:
                        wandb.log({ 
                            "train mean-loss": torch.tensor(losses).mean().item(),
                            "train mean-iou": torch.tensor(ious).mean().item(),
                            "train sub-attn": torch.tensor(subject_attn).mean().item(),
                            "train obj-attn": torch.tensor(object_attn).mean().item(),
                        })
                    else:
                        wandb.log({ 
                            "train mean-loss": torch.tensor(losses).mean().item(),
                            "train mean-iou": torch.tensor(ious).mean().item(),
                        })
        if scheduler is not None:
            scheduler.step()

        logger.critical(f"\nEpoch {epoch} : Mean loss: {sum(losses) / len(losses)}")
        if (epoch + 1) % args.save_interval == 0:
            Path('model_saves/').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict() , os.path.join('model_saves/', args.save_model))

        validation(val_dataset)


def train_unit(lesson, optimizer, scheduler):
    this_train_dataset = train_dataset
    this_val_dataset = val_dataset
    this_train_dataset = this_train_dataset.filter_step(lesson[1])
    this_val_dataset = this_val_dataset.filter_step(lesson[1])
    this_train_dataset = this_train_dataset.filter_scene_size(lesson[2])
    this_val_dataset = this_val_dataset.filter_scene_size(lesson[2])
    if lesson[3] != None:
        this_train_dataset = this_train_dataset.filter_language_complexity(lesson[3])
        this_val_dataset = this_val_dataset.filter_language_complexity(lesson[3])
    if lesson[4]:
        this_train_dataset = this_train_dataset.remove_relational()
        this_val_dataset = this_val_dataset.remove_relational()
    if lesson[5]:
        this_train_dataset = this_train_dataset.only_relational()
        this_val_dataset = this_val_dataset.only_relational()
    this_val_dataset = this_val_dataset.random_trim_length(128)
    mark_unfreezed(model)
    training_loop(train_dataset=this_train_dataset, val_dataset=this_val_dataset, num_epochs=lesson[0] , optimizer=optimizer, scheduler=scheduler)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)    
    
    if args.load_model is not None : 
        logger.critical('Trying to load model from {}'.format(args.load_model))
        try:
            saved_model_state_dict = torch.load(os.path.join('model_saves/', args.load_model))
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except:
            sys.exit("The keys being loaded are incompatible with the current architecture of the model. The model architecture might have changed significantly from the last checkpoint.")
        logger.critical('Loaded the model succesfully!\n')

    if args.load_splitter and args.type == 'multi':
        saved_splitter_embed = torch.load('model_saves/splitter_embed.pth')
        model_embed_dict = model.embed.state_dict()
        pretrained_dict = {k: v for k, v in saved_splitter_embed.items() if k in model_embed_dict}
        model_embed_dict.update(pretrained_dict)
        model.embed.load_state_dict(model_embed_dict)
        
        saved_splitter_lstm = torch.load('model_saves/splitter_lstm.pth')
        model_lstm_dict = model.lstm.state_dict()
        pretrained_dict = {k: v for k, v in saved_splitter_lstm.items() if k in model_lstm_dict}
        model_lstm_dict.update(pretrained_dict)
        model.lstm.load_state_dict(model_lstm_dict)
        
        saved_splitter_linear = torch.load('model_saves/splitter_linear.pth')
        model_linear_dict = model.linear.state_dict()
        pretrained_dict = {k: v for k, v in saved_splitter_linear.items() if k in model_linear_dict}
        model_linear_dict.update(pretrained_dict)
        model.linear.load_state_dict(model_linear_dict)
        
        print("Sentence splitter loaded successfully!")

    if args.type == 'multi':
        optimizer = Adam([
            {"params": model.single_step.language.parameters(), "name": 'parser', "lr": 5e-4},
            {"params": model.single_step.visual.parameters(), "name": 'visual', "lr": 5e-4},
            {"params": model.embed.parameters(), "name": 'embedding', "lr": 3e-4},
            {"params": model.lstm.parameters(), "name": 'lstm', "lr": 5e-4},
            {"params" : model.linear.parameters(), "name": 'linear', "lr": 5e-4},
            {"params": model.single_step.decoder.parameters(), "name": 'decoder', "lr": 9e-4},
            {"params" : model.single_step.subject_map.parameters(), "name": 'subject_map', "lr": 3e-5},
            {"params" : model.single_step.attention_sub.parameters(), "name": 'attention_sub', "lr": 3e-5},
            {"params" : model.single_step.subject_shape_map.parameters(), "name": 'subject_map', "lr": 3e-5},
            {"params" : model.single_step.attention_shape_sub.parameters(), "name": 'attention_shape_sub', "lr": 3e-5},
            {"params" : model.single_step.object_map.parameters(), "name": 'object_map', "lr": 3e-5},
            {"params" : model.single_step.attention_obj.parameters(), "name": 'attention_obj', "lr": 3e-5},
            {"params" : model.single_step.object_shape_map.parameters(), "name": 'object_map', "lr": 3e-5},
            {"params" : model.single_step.attention_shape_obj.parameters(), "name": 'attention_shape_obj', "lr": 3e-5},
            {"params" : model.single_step.pred.parameters(), "name": 'action_sim', "lr": 3e-4}
            ])
    else:
        optimizer = Adam([
            {"params": model.language.parameters(), "name": 'parser', "lr": 5e-4},
            {"params": model.visual.parameters(), "name": 'visual', "lr": 5e-4},
            {"params": model.decoder.parameters(), "name": 'decoder', "lr": 9e-4},
            {"params" : model.subject_map.parameters(), "name": 'subject_map', "lr": 3e-5},
            {"params" : model.object_map.parameters(), "name": 'object_map', "lr": 3e-5},
            {"params" : model.subject_shape_map.parameters(), "name": 'subject_shape_map', "lr": 3e-5},
            {"params" : model.object_shape_map.parameters(), "name": 'object_shape_map', "lr": 3e-5},
            {"params" : model.attention_sub.parameters(), "name": 'attention_sub', "lr": 3e-5},
            {"params" : model.attention_obj.parameters(), "name": 'attention_obj', "lr": 3e-5},
            {"params" : model.attention_shape_sub.parameters(), "name": 'attention_sub', "lr": 3e-5},
            {"params" : model.attention_shape_obj.parameters(), "name": 'attention_obj', "lr" : 3e-5},
            {"params" : model.pred.parameters(), "name": 'action_sim', "lr" : 3e-4}
            ])

    scheduler = StepLR(optimizer, step_size=3, gamma=0.99)

    train_unit([args.num_epochs, args.num_steps, args.num_objects, args.language_complexity, args.remove_relational, args.only_relational], optimizer, scheduler)

    if  args.save_model is not None:
        Path('model_saves/').mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), os.path.join('model_saves/', args.save_model))

    if args.save_splitter and args.type == 'multi':
        Path('model_saves/').mkdir(parents=True, exist_ok=True)
        torch.save(model.embed.state_dict(), 'model_saves/splitter_embed.pth')
        torch.save(model.lstm.state_dict(), 'model_saves/splitter_lstm.pth')
        torch.save(model.linear.state_dict(), 'model_saves/splitter_linear.pth')

        
