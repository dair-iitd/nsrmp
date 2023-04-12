#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   create_condensed_rep.py
#Time    :   2022/07/10 18:05:59
#Author  :   Himanshu G Singh
#Contact :   cs1190358@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

'''
DataLoader class in PyTorch creates an iterable object that creates the batch at runtime.
This slows the feedback cycle for research. Although, it has slight(maybe, no advantage) advantage that the training examples
are shuffled over epochs. 
To rapidly train and test for research purposes, we create the batches once and store them as a pickle file. This can be loaded
into memory straight. 

If it is needed to generate the representation during training/evaluation add the arguments --create_condensed_representation True and 
--use_condensed_representation True
If the pickles are already created, just use the latter
'''

from helpers.logging import get_logger
from helpers.filemanage import ensure_path
from tqdm import tqdm 
import os

logger = get_logger(__file__)

def helper(train_dataset, val_dataset , test_dataset, args):
    import pickle
    dataset_order = ['train','val','test']
    for idx, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
        if dataset is not None:
            logger.critical(f"\n\t\t############### Saving {dataset_order[idx]}.pkl ###################")
            dataloader = dataset.make_dataloader(args.batch_size, shuffle = True, sampler = None, drop_last = True)
            batch_list = []
            for batch in tqdm(dataloader):
                batch_list.append(batch)
            ensure_path("../loader_saves/")
            pickle.dump(batch_list,open(f"../loader_saves/{dataset_order[idx]}.pkl",'wb'))

def create_condensed_representation(args,model_configs):
    from datasets.roboclevr.definition import build_nsrm_dataset
    train_dataset = build_nsrm_dataset(args, model_configs, args.datadir, args.train_scenes_json, args.train_instructions_json, args.vocab_json)
    test_dataset = build_nsrm_dataset(args, model_configs, args.datadir, args.test_scenes_json, args.test_instructions_json, args.vocab_json)
    val_dataset = build_nsrm_dataset(args, model_configs, args.datadir, args.val_scenes_json, args.val_instructions_json, args.vocab_json)
    helper(train_dataset,val_dataset,test_dataset,args)

def condense_dataset(dataset, name:str, batch_size=32, root_dir = '../loader_saves'):
    import pickle
    if dataset is not None:
            print(f"\n\t\t############### Saving {name}.pkl ###################")
            dataloader = dataset.make_dataloader(batch_size, shuffle = True, sampler = None, drop_last = True)
            batch_list = []
            for batch in tqdm(dataloader):
                batch_list.append(batch)
            ensure_path("../loader_saves/")
            pickle.dump(batch_list,open(os.path.join(root_dir,name+'.pkl'),'wb'))