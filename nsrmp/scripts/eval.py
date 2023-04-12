#basic imports
import os.path
import argparse
from tqdm import tqdm
import time
import  matplotlib.pyplot as plt 
import cv2 
import math 

#torch related imports
import torch
from torch.utils.tensorboard import SummaryWriter

#helpers 
from helpers.logging import get_logger, set_log_output_file
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.utils.type_conversion import str2bool
from helpers.utils.container import DOView

#packages within this repo
from datasets.roboclevr.definition import build_nsrm_dataset, build_selfsupervision_dataset
from model.configs import configs as model_configs 

args = argparse.ArgumentParser()

#Added by namas for dataloader
args.add_argument('--dataset', type=str, default='roboclevr')
args.add_argument('--datadir', type = str, default = '../data')
args.add_argument('--train_scenes_json' ,type =str, default ='scenes-test.json')
args.add_argument('--train_instructions_json', type = str, default = 'instructions-test.json')
args.add_argument('--test_instructions_json', type = str, default= 'instructions-test.json')
args.add_argument('--test_scenes_json', type = str, default= 'scenes-test.json')
args.add_argument('--val_instructions_json', type = str, default= 'instructions-test.json')
args.add_argument('--val_scenes_json', type = str, default= 'scenes-test.json')
args.add_argument('--vocab_json', default=None)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--use_cuda', type=str2bool, default = True)
args.add_argument('--training_target', default = 'splitter')
args.add_argument('--instruction_transform', type = str, default = 'basic')

args.add_argument('--load_dir', type = str, required = True, help = 'directory in which the checkpoints are loaded')
args.add_argument('--load_model_from_file',type = str, default=None, help = 'name of the checkpoint to be load')
args.add_argument('--load_modules', nargs='+', type =  str, default=None)


###############################################################

args = args.parse_args()
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
args.train_instructions_json = os.path.join(args.datadir,args.train_instructions_json)
args.val_instructions_json = os.path.join(args.datadir,args.val_instructions_json)
args.test_instructions_json = os.path.join(args.datadir,args.test_instructions_json)
args.train_scenes_json = os.path.join(args.datadir,args.train_scenes_json)
args.val_scenes_json = os.path.join(args.datadir,args.val_scenes_json)
args.test_scenes_json = os.path.join(args.datadir,args.test_scenes_json)

logger = get_logger(__file__)

# logger.critical('Writting logs in /dumps/experiments/{}.log'.format(args.run_name))
# set_log_output_file('../dumps/experiments','{}.log'.format(args.run_name))


#buiding dataset
logger.critical("Building the train, val and test dataset")


train_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'test'), args.train_scenes_json, args.train_instructions_json, args.vocab_json)
val_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'test'), args.val_scenes_json, args.val_instructions_json, args.vocab_json)
test_dataset = build_nsrm_dataset(args, model_configs, os.path.join(args.datadir, 'test'), args.test_scenes_json, args.test_instructions_json, args.vocab_json)

logger.critical("Building the Model")

from model.model_new import Model as Model
model = Model(train_dataset.vocab,model_configs ,training_target = args.training_target)
model.prepare_model('usual')

# train_parser_dataset = build_selfsupervision_dataset(args, model_configs, None, model, 5, args.datadir, create_target = False)
# model.prepare_model("parser_self_supervision")

if args.use_cuda:
    model.cuda()


def test_loop(train_dataset, val_dataset=None, test_dataset=None, **kwargs):
    import pickle 
    logger.critical("Building the dataloader")
    writer = SummaryWriter()

    model.eval()

    dataset_order = ['train', 'val','test'] 
    for idx, dataset in enumerate([train_dataset, val_dataset, test_dataset]): 
        if dataset is not None:
            logger.critical(f"\n\t\t############### Evaluating {dataset_order[idx]}_dataset ###################")
            dataloader = dataset.make_dataloader(args.batch_size, shuffle = True, sampler = None, drop_last = True)
            
            losses = []
            ious = []
            m_obj_ious = []
            b_obj_ious = []
            m_obj_accs = []
            b_obj_accs = []
            m_obj_soft_precision = []
            m_obj_hard_precision = []
            iou_move_objs_sqs = []
            action_accs = []

            for batch in tqdm(dataloader):
                if args.use_cuda:
                    batch = async_copy_to(batch,dev = 0, main_stream=None)

                with torch.no_grad():
                    outputs = model(batch,  **kwargs)
                    
                loss, iou, iou_detail, g_prog_acc = outputs['loss'], outputs['mean_ious'], outputs['indivudual_obj_ious'], outputs['grounded_program_accuracy']
                losses.append(loss.item())
                ious.append(iou.mean().item())
                m_obj_ious.append(iou_detail[0].item())
                b_obj_ious.append(iou_detail[1].item())
                b_obj_accs.append(g_prog_acc[1].item())
                m_obj_accs.append(g_prog_acc[0].item())
                m_obj_soft_precision.append(outputs['move_obj_sp'])
                m_obj_hard_precision.append(outputs['move_obj_hp'])
                action_accs.append(outputs['action_accuracy'])
                iou_move_objs_sqs.append(outputs['move_obj_iou_sqrs'])
            logger.critical(f"Mean loss: {sum(losses)/len(losses)}")
            logger.critical(f"Mean loss: {sum(losses)/len(losses)}")
            logger.critical(f"IOU: min_iou = {min(ious)}, max_iou = {max(ious)}, mean_iou = {sum(ious)/len(ious)}")
            logger.critical(f"Move obj iou = {sum(m_obj_ious)/len(m_obj_ious)}, Base obj iou = {sum(b_obj_ious)/len(b_obj_ious)}")
            logger.critical(f"Move obj iou std = {math.sqrt(sum(iou_move_objs_sqs)/len(iou_move_objs_sqs) - (sum(m_obj_ious)/len(m_obj_ious))**2)}")
            logger.critical(f"Mean action accs: {sum(action_accs)/len(action_accs)}")
            logger.critical(f"Mean Move obj soft Precision (for Cliport): {sum(m_obj_soft_precision)/len(m_obj_soft_precision)}")
            logger.critical(f"Mean Move obj hard Precision (for Cliport): {sum(m_obj_hard_precision)/len(m_obj_hard_precision)}")
            logger.critical(f"Accuracy: Move obj Acc= {sum(m_obj_accs)/len(m_obj_accs)}, Base obj Acc = {sum(b_obj_accs)/len(b_obj_accs)}")
    writer.flush()


if __name__ == '__main__':

    if not args.load_model_from_file == None : 
        from helpers.mytorch.base.serialization import load_state_dict
        load_state_dict(model, os.path.join(args.load_dir, args.load_model_from_file), partial = True, modules = args.load_modules)

    kwargs = dict(unique_mode = 'argmax', parser_self_supervision= False)

    ##single
    #test_dataset = test_dataset.filter_step((0,1))
    ##Double
    #test_dataset = test_dataset.filter_step((0,1))
    ##simple
    #test_dataset = test_dataset.filter_nsrm_program_size((0,8))
    ##complex
    #test_dataset = test_dataset.filter_nsrm_program_size((8,11))
    test_loop(train_dataset = None, val_dataset = None, test_dataset = test_dataset, **kwargs)
    
