#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   dataset.py
#Time    :   2022/06/10 21:03:03
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release


from dis import Instruction
from nltk.tokenize import word_tokenize
import numpy as np
import random
import os
from collections import defaultdict


import torch
import torch.nn.functional as F

from helpers.logging import get_logger
from helpers.utils.container import DOView
import helpers.io as io 
from helpers.filemanage import ensure_path


from .roboclevr.program_transforms import *
from datasets.common.collate import SimpleCollate
from nsrmp.datasets.vocab import Vocab
from datasets.definition import gdef
from datasets.common.program_translator import nsrmseq_to_nsrmqsseq, nsrmseq_to_nsrmtree, nsrmtree_to_nsrmqstree, nsrmtree_to_nsrmseq
from datasets.common.filterable import FilterableDatasetBase, FilterableDataset


logger = get_logger(__file__)

class MPLDatasetUnwrapped(FilterableDatasetBase):
    def __init__(self,args, scenes_json, instructions_json, demo_root, image_transform, instruction_transform, vocab_json = None, **kwargs):
        super().__init__()
        self.scenes_json = scenes_json
        self.instructions_json = instructions_json
        self.demo_root = demo_root
        self.image_transform = image_transform
        self.instruction_transform = instruction_transform
        self.vocab_json = vocab_json

        logger.info('Loading Scenes from: {}'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        self.initial_scenes = [self.scenes[i][0] for i in range(len(self.scenes))]
        self.final_scenes = [self.scenes[i][1] for i in range(len(self.scenes))]
        logger.info("Loading instructions from: {}".format(self.instructions_json))
        self.instructions = io.load_json(self.instructions_json)["instructions"]
        if self.vocab_json is None:
            logger.critical('generating vocab from the dataset')
            from datasets.definition import gdef
            self.vocab = Vocab.from_dataset(self.get_sent,dataset_length = len(self) ,extra_words=gdef.extra_embeddings,vocab_cls = Vocab, save_vocab=args.save_vocab)
        else:
            logger.info("Loading Vocab from: {}".format(self.vocab_json))
            self.vocab = Vocab.from_json(self.vocab_json)
        
        for k,value in kwargs.items():
            setattr(self,k,value)
            
    def _get_metainfo(self,index):
        instruction = self.instructions[index]
        info = {}
        info['initial_scene'] = self.initial_scenes[index]
        info['final_scene'] = self.final_scenes[index]
        info['grounded_program'] = instruction['grounded_program']
        if 'program' in instruction:
            info['program_raw'] = instruction["program"]
            info['program_seq'] = gdef.program_to_nsrmseq(instruction['program'])
            info['program_tree'] = nsrmseq_to_nsrmtree(info['program_seq'])
            info['program_qsseq'] = nsrmseq_to_nsrmqsseq(info['program_seq'])
            info['program_qstree'] = nsrmtree_to_nsrmqstree(info['program_tree'])

        info['instruction_raw'] = word_tokenize(self.instructions[index]['instruction'])
        info['language_complexity'] = instruction['language_complexity']
        return info

    def get_sent(self,index):
        return word_tokenize(self.instructions[index]['instruction'])

    def __getitem__(self,index):
        metainfo = DOView(self.get_metainfo(index))
        sample = metainfo
        sample.initial_image, sample.initial_bboxes = self.image_transform(metainfo.initial_scene)
        sample.final_image, sample.final_bboxes = self.image_transform(metainfo.final_scene)
        sample.object_length = sample.initial_bboxes.size(0)
        if self.instruction_transform is not None:
            tokens,out_dict = self.instruction_transform(metainfo.instruction_raw)
            sample.instruction = np.array([self.vocab.word2idx.get(w,self.vocab.word2idx['<UNK>']) for w in tokens],dtype='int64')
            sample.update(out_dict)
        return sample.make_dict()

    def __len__(self):
        return len(self.instructions)


class MPLDatasetFilterable(FilterableDataset):
    def filter_scene_size(self, scene_size_range):
        min_size, max_size = scene_size_range
        def filt(sample):
            return len(sample['initial_scene']['objects']) > min_size and len(sample['initial_scene']['objects']) <= max_size
        return self.filter(filt, 'filter-scene-size({})'.format((min_size,max_size)))
    
    def filter_instruction():
        raise NotImplementedError
    
    def filter_questions():
        raise NotImplementedError
   
    def filter_step(self, step_range):
        min_step,max_step = step_range
        def filt(sample):
            return len(sample['grounded_program']) > min_step and len(sample['grounded_program']) <= max_step
        return self.filter(filt, 'filter-num-action[{}]'.format((min_step,max_step)))

    def filter_program_raw_size(self, length_range):
        min_len, max_len = length_range
        def filt(sample):
            return len(sample['program_raw']) > min_len  and len(sample['program_raw']) <= max_len 
        return self.filter(filt, 'filter-program-raw-size[{}]'.format((min_len,max_len)))
    
    def filter_nsrm_program_size(self, length_range):
        min_len, max_len = length_range
        def filt(sample):
            return len(sample['program_qsseq']) > min_len and len(sample['program_qsseq']) <= max_len
        return self.filter(filt, 'filter-program-raw-size[{}]'.format((min_len,max_len)))

    def filter_language_complexity(self, type):
        def filt(sample):
            return sample['language_complexity'] == type
        return self.filter(filt, 'filter_language_complexity{}'.format(type))
   
    def make_dataloader(dataset,batch_size,shuffle=True,sampler = None, drop_last = True):
        from torch.utils.data import DataLoader
        collate_guide = {
            'instruction':'pad',
            'initial_image':'basic',
            'final_image' :'basic',
             'object_length':'basic'
        }
        return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,sampler=sampler, drop_last= drop_last, collate_fn=NSRMCollate(collate_guide))



def MPLDataset(*args, filter_cls = MPLDatasetFilterable, **kwargs):
    return filter_cls(NSRMDatasetUnwrapped(*args,**kwargs))



class NSRMDatasetUnwrapped(FilterableDatasetBase):
    def __init__(self,args, scenes_json, instructions_json, demo_root, image_transform, instruction_transform, vocab_json = None, **kwargs):
        super().__init__()
        self.scenes_json = scenes_json
        self.instructions_json = instructions_json
        self.demo_root = demo_root
        self.image_transform = image_transform
        self.instruction_transform = instruction_transform
        self.vocab_json = vocab_json

        logger.info('Loading Scenes from: {}'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)['scenes']
        self.initial_scenes = [self.scenes[i][0] for i in range(len(self.scenes))]
        self.final_scenes = [self.scenes[i][1] for i in range(len(self.scenes))]
        logger.info("Loading instructions from: {}".format(self.instructions_json))
        self.instructions = io.load_json(self.instructions_json)["instructions"]
        if self.vocab_json is None:
            logger.critical('generating vocab from the dataset')
            from datasets.definition import gdef
            self.vocab = Vocab.from_dataset(self.get_sent,dataset_length = len(self) ,extra_words=gdef.extra_embeddings,vocab_cls = Vocab, save_vocab=args.save_vocab)
        else:
            logger.info("Loading Vocab from: {}".format(self.vocab_json))
            self.vocab = Vocab.from_json(self.vocab_json)
        
        for k,value in kwargs.items():
            setattr(self,k,value)
            
    def _get_metainfo(self,index):
        instruction = self.instructions[index]
        info = {}
        info['initial_scene'] = self.initial_scenes[index]
        info['final_scene'] = self.final_scenes[index]
        info['grounded_program'] = instruction['grounded_program']
        if 'program' in instruction:
            info['program_raw'] = instruction["program"]
            info['program_seq'] = gdef.program_to_nsrmseq(instruction['program'])
            info['program_tree'] = nsrmseq_to_nsrmtree(info['program_seq'])
            info['program_qsseq'] = nsrmseq_to_nsrmqsseq(info['program_seq'])
            info['program_qstree'] = nsrmtree_to_nsrmqstree(info['program_tree'])

        info['instruction_raw'] = self.instructions[index]['instruction']
        info['instruction_template'] = self.instructions[index]['instruction_lexed'].split()
        info['metadata'] = dict(template_id = self.instructions[index]['template_id'], template_json_filename = self.instructions[index]['template_json_filename'])
        info['language_complexity'] = instruction['language_complexity']
        return info

    def get_sent(self,index):
        return word_tokenize(self.instructions[index]['instruction'])

    def __getitem__(self,index):
        metainfo = DOView(self.get_metainfo(index))
        sample = metainfo
        sample.initial_image, sample.initial_bboxes = self.image_transform(metainfo.initial_scene)
        sample.final_image, sample.final_bboxes = self.image_transform(metainfo.final_scene)
        sample.object_length = sample.initial_bboxes.size(0)
        if self.instruction_transform is not None:
            tokens,out_dict = self.instruction_transform(word_tokenize(metainfo.instruction_raw), metainfo.instruction_template)
            sample.instruction_raw = metainfo.instruction_raw
            sample.instruction_lexed = '<BOS> ' + ' '.join(tokens) + ' <EOS>'
            sample.instruction = np.array([self.vocab.word2idx['<BOS>']]+[self.vocab.word2idx.get(w,self.vocab.word2idx['<UNK>']) for w in tokens] + [self.vocab.word2idx['<EOS>']],dtype='int64')
            sample.update(out_dict)
        else:
            sample.instruction = np.array([self.vocab.word2idx['<BOS>']]+[self.vocab.word2idx.get(w,self.vocab.word2idx['<UNK>']) for w in word_tokenize(metainfo.instruction_raw)] + [self.vocab.word2idx['<EOS>']],dtype='int64')
        return sample.make_dict()

    def __len__(self):
        return len(self.instructions)



class NSRMDatasetFilterable(FilterableDataset):
    def filter_scene_size(self, scene_size_range):
        min_size, max_size = scene_size_range
        def filt(sample):
            return len(sample['initial_scene']['objects']) > min_size and len(sample['initial_scene']['objects']) <= max_size
        return self.filter(filt, 'filter-scene-size({})'.format((min_size,max_size)))
    
    def filter_instruction():
        raise NotImplementedError
    
    def filter_questions():
        raise NotImplementedError
    
    def filter_actions(self,actions):
        def filt(sample):   
            all_actions = [s[0] for s in sample['grounded_program']]
            for a in all_actions:
                if a not in actions:
                    return False
            return True
        return self.filter(filt, "filter-actions[{}]".format(actions))
   
    def filter_step(self, step_range):
        min_step,max_step = step_range
        def filt(sample):
            return len(sample['grounded_program']) > min_step and len(sample['grounded_program']) <= max_step
        return self.filter(filt, 'filter-num-action[{}]'.format((min_step,max_step)))

    def filter_program_raw_size(self, length_range):
        min_len, max_len = length_range
        def filt(sample):
            return len(sample['program_raw']) > min_len  and len(sample['program_raw']) <= max_len 
        return self.filter(filt, 'filter-program-raw-size[{}]'.format((min_len,max_len)))
    
    def filter_nsrm_program_size(self, length_range):
        min_len, max_len = length_range
        def filt(sample):
            return len(sample['program_qsseq']) > min_len and len(sample['program_qsseq']) <= max_len
        return self.filter(filt, 'filter-program-raw-size[{}]'.format((min_len,max_len)))

    def filter_language_complexity(self, type):
        def filt(sample):
            return sample['language_complexity'] == type
        return self.filter(filt, 'filter_language_complexity [{}]'.format(type))

    def filter_templates(self, template_ids):
        def filt(sample):
            return sample['metadata']['template_id'] in template_ids
        return self.filter(filt, 'filter_templates [{}]'.format(template_ids))
    
    def remove_relational(self):
        def filt(sample):
            for d in sample['program_raw']:
                if d['type'] == 'relate':
                    return False 
            return True 
        return self.filter(filt,'remove-relational')

    def filter_relational(self):
        def filt(sample):
            for d in sample['program_raw']:
                if d['type'] == 'relate':
                    return True
            return False
        return self.filter(filt,'filter-relational')
        
        
    def make_dataloader(dataset,batch_size,shuffle=True,sampler = None, drop_last = True):
        from torch.utils.data import DataLoader
        collate_guide = {
            'instruction':'pad',
            'initial_image':'basic',
            'final_image' :'basic',
             'object_length':'basic'
        }
        return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,sampler=sampler, drop_last= drop_last, collate_fn=NSRMCollate(collate_guide))


def NSRMDataset(*args, filter_cls = NSRMDatasetFilterable, **kwargs):
    return filter_cls(NSRMDatasetUnwrapped(*args,**kwargs))


class NSRMCollate(SimpleCollate):
    def __init__(self, collate_guide):
        super().__init__(collate_guide)
    
    def __call__(self,batch):
        batched_dict = super().__call__(batch)
        return batched_dict


class NSRMParserSelfSupervisionDataset(FilterableDatasetBase):
    def __init__(self, owner_dataset, model, max_steps, root_dir, program_transform, instruction_transform, target_dataset_name = None, create_target = False, **kwargs):
        super().__init__()
        self.parent_dataset = owner_dataset
        self.model = model
        self.max_steps = max_steps
        self.root_dir = root_dir
        self.target_dataset_name = target_dataset_name if target_dataset_name is not None else f"parser_{max_steps}_dataset"
        self.instruction_transform = instruction_transform
        self.program_transform = program_transform
        self.instructions = None
        if create_target == False:
            self.instructions = io.load_json(os.path.join(self.root_dir, self.target_dataset_name+'.json'))["instructions"]
        self.use_groundtruth = kwargs.get('use_groundtruth', False)
    
    def _get_metainfo(self, index):
        if self.instructions is None:
            return self.parent_dataset[index]
        else: 
            return self.__getitem__(index)

    def __getitem__(self, index=None):
        if index is None:
            
            num_steps = random.randint(1, self.max_steps)
            this_indices = random.sample(range(len(self.parent_dataset)), num_steps)
            this_metadatas = [self._get_metainfo(i) for i in this_indices]
            sample = DOView()
            this_instructions_raw = [metadata['instruction_raw'].strip() for metadata in this_metadatas]
            this_instructions_lexed = [metadata['instruction_lexed'].strip() for metadata in this_metadatas]
            tokens_raw, tokens_lexed = self.instruction_transform(this_instructions_raw, this_instructions_lexed)
            
            sample.instruction_lexed = '<BOS> ' + ' '.join(tokens_lexed) + ' <EOS>'
            sample.instruction_raw = ' '.join(tokens_raw)
            sample.instruction = [self.parent_dataset.vocab.word2idx['<BOS>']] + [self.parent_dataset.vocab.word2idx.get(w,self.parent_dataset.vocab.word2idx['<UNK>']) for w in tokens_lexed] + [self.parent_dataset.vocab.word2idx['<EOS>']]           
            concept_dict = defaultdict(list)
            for data in this_metadatas:
                for k in gdef.concept_groups:
                    concept_dict[k].extend(data[k])
            sample.update(concept_dict)
            
            with torch.no_grad():
                if self.use_groundtruth:
                    sample.program , _,_,_ = self.program_transform([data['program_qstree'] for data in this_metadatas],
                    [data['attribute_concepts'] for data in this_metadatas],
                    [data['relational_concepts'] for data in this_metadatas],
                    [data['action_concepts'] for data in this_metadatas],
                    )
                else:
                    sample.program , _,_,_ = self.program_transform([nsrmseq_to_nsrmtree(self.model.parser(torch.tensor(data['instruction']).unsqueeze(0), torch.tensor([len(data['instruction'])]), 12, [data['attribute_concepts']],
                                                                            [data['relational_concepts']], [data['action_concepts']],
                                                                            sample_method='sample', sample_space=None, exploration_rate = 0)[0]['program'])
                                                            for data in this_metadatas],
                    [data['attribute_concepts'] for data in this_metadatas],
                    [data['relational_concepts'] for data in this_metadatas],
                    [data['action_concepts'] for data in this_metadatas],
                    )

            return sample.make_dict()
        else:
            assert self.instructions is not None, "Please create the dataset by calling _create_and_dump() before using the dataset"
            sample = DOView()
            this_sample = self.instructions[index]
            sample.instruction = torch.tensor(this_sample['instruction'], dtype = torch.int64)
            sample.groundtruth_qstree = this_sample['program']
            sample.program_qsseq = nsrmtree_to_nsrmseq(this_sample["program"])
            sample.instruction_lexed = this_sample['instruction_lexed']
            sample.instruction_raw = this_sample["instruction_raw"]
            for k in gdef.concept_groups:
                sample[k] = this_sample[k]
            return sample

    def __len__(self):
        if self.instructions is None:
            raise AttributeError("Please use _create_and_dump() to create the dataset before using it")
        else:
            return len(self.instructions)


    def _create_and_dump(self, length, split):
        logger.critical("Creating the Self supervision parser dataset")
        from datetime import date
        dump = {
        'info':{
            'date': str(date.today()),
            'split': split
        },
        'instructions': []
        }
        from tqdm import tqdm
        for i in tqdm(range(length)):
            dump['instructions'].append(self.__getitem__())
        
        
        import json
        logger.critical(f'Creating {self.target_dataset_name}.json in {self.root_dir}. Overridding the file if already exisits')
        with open(os.path.join(self.root_dir, self.target_dataset_name+'.json'), 'w') as f:
            json.dump(dump, f)

def NSRMParserDataset(*args, filter_cls = NSRMDatasetFilterable, **kwargs):
    return filter_cls(NSRMParserSelfSupervisionDataset(*args,**kwargs))

        
    
