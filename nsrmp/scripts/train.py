#basic imports
import os
import argparse
from tqdm import tqdm
import time
import  matplotlib.pyplot as plt 
import cv2 
from pathlib import Path
import shutil
import pickle, random
import numpy as np

#torch related imports
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets.roboclevr import instruction_transforms
from helpers.filemanage.filemanage import ensure_path

#helpers 
from helpers.logging import get_logger, set_log_output_file
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.mytorch.train.freeze import mark_unfreezed, mark_freezed
from helpers.utils.type_conversion import str2bool 


#packages within this repo
from datasets.roboclevr.definition import build_nsrm_dataset, build_selfsupervision_dataset
from helpers.utils.container import DOView
from model.configs import configs as model_configs 
from scripts.curriculum import get_curr 

args = argparse.ArgumentParser()

args.add_argument('--ensure_reproducibility', type = str2bool, default=True)
args.add_argument('--initialization_save_dir', type = str, default = "../initialization_saves")
args.add_argument('--save_initial_weights', type = str2bool, default = False )

args.add_argument('--dataset', type=str, default='roboclevr')
args.add_argument('--datadir', type = str, required= True)
args.add_argument('--train_scenes_json' ,type =str, default ='scenes-train.json')
args.add_argument('--train_instructions_json', type = str, default = 'instructions-train.json')
args.add_argument('--test_instructions_json', type = str, default= 'instructions-test.json')
args.add_argument('--test_scenes_json', type = str, default= 'scenes-test.json')
args.add_argument('--val_instructions_json', type = str, default= 'instructions-val.json')
args.add_argument('--val_scenes_json', type = str, default= 'scenes-val.json')
args.add_argument('--vocab_json', default=None)
args.add_argument('--save_vocab', type = str2bool, default=False)

args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--num_epochs', type=int, default=2000)
args.add_argument('--use_cuda', type = str2bool, default = True)
args.add_argument('--training_target',type =str, default = 'all')
args.add_argument('--instruction_transform', type = str, default = 'basic', choices=['off', 'basic', 'program_parser_candidates'])
args.add_argument('--eval_interval',type = int, default=10)

args.add_argument('--model_save_interval', type = int, default = 100)
args.add_argument('--model_save_dir',type = str, default="../model_saves")
args.add_argument('--save_model_to_file', type = str, default='None')

args.add_argument('--load_dir', type = str, default = None, help = 'directory in which the checkpoints are loaded')
args.add_argument('--load_model_from_file',type = str, default=None, help = 'name of the checkpoint to be load')
args.add_argument('--load_modules', nargs='+', type =  str, default=None)

args.add_argument('--wandb',type = str2bool, default=False, help= "visualize and see the meters in weight and biases")
args.add_argument('--use_condensed_representation',type=str2bool,default=False, help= "Enable pickling the data to speed up training")


args = args.parse_args()

args.train_instructions_json = os.path.join(args.datadir,args.train_instructions_json)
args.val_instructions_json = os.path.join(args.datadir,args.val_instructions_json)
args.test_instructions_json = os.path.join(args.datadir,args.test_instructions_json)
args.train_scenes_json = os.path.join(args.datadir,args.train_scenes_json)
args.val_scenes_json = os.path.join(args.datadir,args.val_scenes_json)
args.test_scenes_json = os.path.join(args.datadir,args.test_scenes_json)

args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d---%H-%M-%S'))

logger = get_logger(__file__)

# logger.critical('Writting logs in /dumps/experiments/{}.log'.format(args.run_name))
# set_log_output_file('../dumps/experiments','{}.log'.format(args.run_name))

# logger.critical(f"\nArguments passed :\n {vars(args)}")

if args.ensure_reproducibility:
    TORCH_SEED = 1
    torch.manual_seed(TORCH_SEED)
    logger.critical(f"Setting the seed for torch: TORCH_SEED =  {TORCH_SEED}")

    NUMPY_SEED = 1
    np.random.seed(NUMPY_SEED)
    logger.critical(f"Setting the seed for numpy: NUMPY_SEED = {NUMPY_SEED}")

    PYTHON_RND_SEED = 1
    random.seed(PYTHON_RND_SEED)
    logger.critical(f"Setting the seed for python random: PYTHON_RND_SEED = {NUMPY_SEED}")

#buiding dataset
logger.critical("Building the train and val dataset")

train_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'train'), args.train_scenes_json, args.train_instructions_json, args.vocab_json)
val_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'val'), args.val_scenes_json, args.val_instructions_json, args.vocab_json, instruction_transform = 'basic')


logger.critical("Building the Model")
from model.model_new import Model as Model
model = Model(train_dataset.vocab, model_configs ,training_target = args.training_target,)
model.prepare_model('usual')
logger.critical(f"Configs of model is \n{model_configs.data.make_dict()}, \n{model_configs.model.make_dict()}, \n{model_configs.model.parser.make_dict()}")

if args.use_cuda:
    model.cuda()

if args.ensure_reproducibility:
    if args.save_initial_weights:
        init_weight_filename = os.path.join(args.initialization_save_dir,args.run_name+'.pth')
        ensure_path(args.initialization_save_dir)
        logger.critical(f'Saving the initial_weights to {init_weight_filename}')
        torch.save(model.state_dict(), init_weight_filename)
        if args.load_model_from_file is not None:
            logger.warning(f"The model is being loded from {args.load_model_from_file}. The initial weights may change.")

if args.wandb:
    import wandb
    wandb.config = {
    "lr" : 0.01,
    "batch_size" : 32,
    } 
    wandb.init(project="nsrmp", entity="nsrmp")
    wandb.watch(model, log='all', log_freq=1)

def training_loop(train_dataset, val_dataset, num_epochs, optimizer, scheduler = None, **kwargs):
    writer = SummaryWriter()
    for epoch in range(num_epochs):
 
        model.train(freeze_modules = kwargs.get('freeze_modules',None))
       
        if args.use_condensed_representation:
            #The dataset is pickled beforehand and the same is 75% of the times to increase the speed of training.
            rnd = random.random()
            if rnd< 0.75:
                train_dataloader = pickle.load(open(f"../loader_saves/train.pkl",'rb'))
                random.shuffle(train_dataloader)
            else:
                train_dataloader = train_dataset.make_dataloader(args.batch_size, shuffle = True, sampler = None, drop_last = True)
        else:
            train_dataloader = train_dataset.make_dataloader(args.batch_size, shuffle = True, sampler = None, drop_last = True)

        losses = []
        ious = []
        m_obj_ious = []
        b_obj_ious = []
        m_obj_accs = []
        b_obj_accs = []

        # start_time = time.time()
        pbar = tqdm(enumerate(train_dataloader))
        for iteration,batch in pbar:

            # print("data:", time.time()-start_time)
            # start_time = time.time()
            if args.use_cuda:
                batch = async_copy_to(batch,dev = 0, main_stream=None)
            # print("async copy: ",time.time()-start_time)
            # start_time = time.time()
            try:
                outputs = model(batch,  **kwargs)
            except Exception as e:
                logger.error(f'\nError in epoch {epoch} batch {iteration}', exc_info = True, stack_info = True)
            
            # print("model:", time.time()-start_time)

            if not model.training_target == 'splitter':
                # start_time = time.time()
                loss, iou, iou_detail, g_prog_acc = outputs['loss'], outputs['mean_ious'], outputs['indivudual_obj_ious'], outputs['grounded_program_accuracy']
                optimizer.zero_grad()
                loss.backward()
                # print("bkkwd:", time.time()-start_time)
                optimizer.step()
                # start_time = time.time()
                losses.append(loss.item())
                ious.append(iou.mean().item())
                m_obj_ious.append(iou_detail[0].item())
                b_obj_ious.append(iou_detail[1].item())
                b_obj_accs.append(g_prog_acc[1].item())
                m_obj_accs.append(g_prog_acc[0].item())
                pbar.set_description(f'Epoch: {epoch}, Loss:{loss.item()}, Move iou: {iou_detail[0].item()}, Move_obj_acc: {g_prog_acc[0].item()}, Base_obj_acc: {g_prog_acc[1].item()}' )
            else:
                loss = outputs['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.set_description(f'Epoch: {epoch}, Loss:{sum(losses)/len(losses)}')
                
                if iteration%4 ==0:
                    if args.wandb:
                        wandb.log({
                            'train mean-loss': torch.tensor(losses).mean().item(),
                        })

        if scheduler is not None:
            scheduler.step()

        if (epoch+1)%args.model_save_interval == 0:
            ensure_path(args.model_save_dir)
            if  args.save_model_to_file is not None:
                torch.save(model.state_dict(),os.path.join(args.model_save_dir,args.save_model_to_file))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_save_dir,'model_checkpoint.pth'))
            
            
        logger.critical(f"\nEpoch {epoch} : Mean loss: {sum(losses)/len(losses)} \n")
        if not model.training_target == 'splitter':
            logger.critical("IOUS: min_iou = {min(ious)}, max_iou = {max(ious)}, mean_iou = {sum(ious)/len(ious)}, \nINDIVIDUAL IOU: Move obj iou = {sum(m_obj_ious)/len(m_obj_ious)}, Base obj iou = {sum(b_obj_ious)/len(b_obj_ious)}, \nACCURACY: Move obj Acc= {sum(m_obj_accs)/len(m_obj_accs)}, Base obj Acc = {sum(b_obj_accs)/len(b_obj_accs)} \n")
        
        if not model.training_target == 'splitter' and args.wandb:
            wandb.log({
                'train mean-loss': torch.tensor(losses).mean().item(),
                'train mean-iou': torch.tensor(ious).mean().item(),
                'train move-obj-iou': sum(m_obj_ious)/len(m_obj_ious),
                'train program-acc': (sum(b_obj_accs)+sum(m_obj_accs))/(2*len(b_obj_accs))
            }) 
        
        if (epoch+1)%args.eval_interval == 0:
            if val_dataset is not None:
                model.eval()
                logger.info("\n####### Validating the model #########")
                validation(val_dataset, parser_only= model.training_target == 'splitter')
            
    writer.flush()

def validation(val_dataset,name="validation", parser_only = False, **kwargs):
    val_dataloader = val_dataset.make_dataloader(args.batch_size, shuffle = True, sampler = None, drop_last = True)
    losses = []
    if not parser_only:
        ious = []
        m_obj_ious = []
        b_obj_ious = []
        m_obj_accs = []
        b_obj_accs = []
        

    for batch in tqdm(val_dataloader):
        if args.use_cuda:
            batch = async_copy_to(batch,dev = 0, main_stream=None)
        with torch.no_grad():
            outputs = model(batch,  unique_mode = 'argmax', **kwargs)
        loss = outputs['loss']
        losses.append(loss.item())
        if not parser_only:
            iou, iou_detail, g_prog_acc = outputs['mean_ious'], outputs['indivudual_obj_ious'], outputs['grounded_program_accuracy']
            ious.append(iou.mean().item())
            m_obj_ious.append(iou_detail[0].item())
            b_obj_ious.append(iou_detail[1].item())
            b_obj_accs.append(g_prog_acc[1].item())
            m_obj_accs.append(g_prog_acc[0].item())
        else: 
            loss = outputs['loss']
            losses.append(loss)
          
    
    logger.critical(f"Mean loss: {sum(losses)/len(losses)}")

    if not parser_only:
        logger.critical(f"IOU: min_iou = {min(ious)}, max_iou = {max(ious)}, mean_iou = {sum(ious)/len(ious)}")
        logger.critical(f"INDIVIDUAL IOU: Move obj iou = {sum(m_obj_ious)/len(m_obj_ious)}, Base obj iou = {sum(b_obj_ious)/len(b_obj_ious)}")
        logger.critical(f"ACCURACY: Move obj Acc= {sum(m_obj_accs)/len(m_obj_accs)}, Base obj Acc = {sum(b_obj_accs)/len(b_obj_accs)}\n")
    
    if args.wandb:
        if not parser_only:
            wandb.log({
            f'{name} mean-loss': torch.tensor(losses).mean().item(),
            f'{name} mean-iou': torch.tensor(ious).mean().item(),
            f'{name} move-obj-iou': sum(m_obj_ious)/len(m_obj_ious),
            f'{name}  program-acc': (sum(b_obj_accs)+sum(m_obj_accs))/(2*len(b_obj_accs))
            }) 
        else:
            wandb.log({
            f'{name} mean-loss': torch.tensor(losses).mean().item(),
            }) 



def train_unit(**unit):
    lesson = unit.get('lesson')
    optimizer = unit.get("optimizer")
    scheduler = unit.get("scheduler", None)
    
    if unit.get('change_group_concepts', None):
        logger.critical(f"changing grouping concepts from {model_configs.data.group_concepts} to {unit['change_group_concepts']}")
        this_train_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'train'), args.train_scenes_json, args.train_instructions_json, args.vocab_json, group_concepts = unit['change_group_concepts'])
    elif unit.get('change_instruction_transform', None):
        logger.critical(f"changing instruction transform from {args.instruction_transform} to {unit['change_instruction_transform']}")
        this_train_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'train'), 
                                                args.train_scenes_json, args.train_instructions_json, args.vocab_json,  instruction_transform = unit['change_instruction_transform']) 
    else:
        this_train_dataset = train_dataset
    
    this_val_dataset = val_dataset
    
    if lesson.get('num_step_range', None):
        this_train_dataset = this_train_dataset.filter_step(lesson['num_step_range'])
        # this_val_dataset = this_val_dataset.filter_step(lesson['num_step_range'])
    if lesson.get('scene_size_range', None):
        this_train_dataset = this_train_dataset.filter_scene_size(lesson['scene_size_range'])
        # this_val_dataset = this_val_dataset.filter_scene_size(lesson['scene_size_range'])
    if lesson.get('program_size_range', None):
        this_train_dataset = this_train_dataset.filter_nsrm_program_size(lesson['program_size_range'])
        # this_val_dataset = this_val_dataset.filter_nsrm_program_size(lesson['program_size_range'])
    if lesson.get('language_complexity', None):
        this_train_dataset = this_train_dataset.filter_language_complexity(lesson['language_complexity'])
        # this_val_dataset = this_val_dataset.filter_language_complexity(lesson['language_complexity'])

    if lesson.get('length',None):
        this_train_dataset = this_train_dataset.random_trim_length(lesson['length'])
        this_val_dataset = this_val_dataset.random_trim_length(lesson['length']//32)  

    if args.use_condensed_representation:
        from create_condensed_rep import condense_dataset
        condense_dataset(this_train_dataset,'train', batch_size=args.batch_size,)

    logger.critical(f"Creating curriculum. details: {lesson}, len: {len(this_train_dataset)}")
    
    mark_unfreezed(model)
    if unit.get('change_training_target', None):
        training_target = unit['change_training_target']
        logger.critical(f"Changing training target from {model.training_target} to {training_target}")
        model.training_target = training_target
        if unit.get('reset_target', False):
            logger.critical(f"Request received to reset the parameters of the training target: {training_target}. Trying to perform the reset!")
            model.reset(training_target)
        model.prepare_model('usual')

    kwargs = {k:unit.get(k) for k in ['unique_mode', 'gumbel_tau', 'freeze_modules'] if k in unit}
    kwargs.update(unit.get("extras",{}))

    if args.training_target == 'splitter' or args.training_target == 'action_simulator':
        kwargs['unique_mode'] = 'argmax'

    logger.info(f"\ncalling the training loop. Curriculum = {lesson}, num_epochs: {lesson['num_epochs']},  kwargs = {kwargs}, optimizer = {optimizer}\n")
    training_loop(this_train_dataset, val_dataset=this_val_dataset, num_epochs = lesson['num_epochs'] ,optimizer = optimizer, scheduler = scheduler, **kwargs)
    
    if unit.get('change_training_target', None):
        logger.critical(f"Changing the training target back to the one passed in args. i.e. {args.training_target}")
        model.training_target = args.training_target
        model.prepare_model('usual')
    
    
if __name__ == '__main__':

    if not args.load_model_from_file == None : 
        load_dir = args.load_dir if args.load_dir is not None else args.model_save_dir
        from helpers.mytorch.base.serialization import load_state_dict
        load_state_dict(model, os.path.join(load_dir, args.load_model_from_file), partial = True, modules = args.load_modules)

    curriculum = get_curr(model)
    from helpers.mytorch.train.freeze import mark_unfreezed, mark_freezed

    total_epochs_so_far =  0
    for i,unit in enumerate(curriculum):
        if total_epochs_so_far > args.num_epochs:
            break  
        train_unit(**unit)
        total_epochs_so_far += unit['lesson']['num_epochs']

    if  args.save_model_to_file is not None:
        ensure_path(args.model_save_dir)
        torch.save(model.state_dict() ,os.path.join(args.model_save_dir,args.save_model_to_file))
        if args.wandb:
            Path(wandb.run.dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, args.save_model_to_file) )

    if args.wandb:
        Path(wandb.run.dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join('../dumps/experiments',f'{args.run_name}.log'), os.path.join(wandb.run.dir,f'{args.run_name}.log'))



