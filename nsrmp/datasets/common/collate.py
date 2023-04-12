#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   collate.py
#Time    :   2022/06/10 21:03:23
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release



import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class SimpleCollate(object):
    def __init__(self,guide):
        self.guide = guide  

    def collate(self,batch,key):
        collate_option = self.guide.get(key,'skip')
        if collate_option == 'skip':
            return  {key:batch}

        elif collate_option == 'basic':
            if type(batch[0]).__module__ == np.__name__:
                    batch = [torch.tensor(sample) for sample in batch]
            if torch.is_tensor(batch[0]):
                batch =  torch.stack(batch)
            elif type(batch[0]) == int or float:
                batch = torch.tensor(batch)
            else:
                raise NotImplementedError
            return {key:batch}

        elif collate_option == 'pad':
            for sample in batch: 
                assert sample.ndim == 1, 'illegal dim of ndarray for collate_option pad. Expected {} but got {}'.format(1, sample.ndim)
            if type(batch[0]).__module__ == np.__name__:
                batch = [torch.tensor(sample) for sample in batch]
            if torch.is_tensor(batch[0]):
                lengths = torch.tensor([sample.shape[0] for sample in batch],dtype=torch.int64)
                padded = pad_sequence(batch,batch_first=True, padding_value=0)
                return {key:padded, str(key)+'_length':lengths}
            else:
                raise NotImplementedError

        elif collate_option == 'pad2d':
            for sample in batch:
                assert sample.ndim == 2, 'iilegal dim of ndarray for collate option pad2d. Expected {} but got {}'.format(2, sample.ndim)
            # as of now, the padding techinque for pad2d is same as that of pad. But this can change in future commits
            if type(batch[0]).__module__ == np.__name__:
                    batch = [torch.tensor(sample) for sample in batch]
            if torch.is_tensor(batch[0]):
                lengths = torch.tensor([sample.shape[0] for sample in batch],dtype=torch.int64)
                padded = pad_sequence(batch,batch_first=True, padding_value=0).type(torch.FloatTensor)
                return {key:padded, str(key)+'_length':lengths}
            else:
                raise NotImplementedError

        elif collate_option == 'padimage':
            raise NotImplementedError

    def __call__(self,batch:list):
        #if some key is not present in guide, the default collate option is skip
        keys = batch[0].keys()
        batched_dict = {}
        for k in keys:
            batched_dict.update(self.collate([sample[k] for sample in batch], k))
        return batched_dict


    