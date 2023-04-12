#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   filterable.py
#Time    :   2022/07/20 19:41:04
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

#Acknowledgement: Parts of this implementation is adopted from NSCL PyTorch Release

import random 
from copy import deepcopy
from torch.utils.data.dataset import Dataset
from helpers.logging import get_logger

logger = get_logger(__file__)


class FilterableDatasetBase(Dataset):
    """
    A filterable dataset. User can call various `filter_*` operations to obtain a subset of the dataset.
    """
    def __init__(self):
        super().__init__()
        self.metainfo_cache = dict()

    def get_metainfo(self, index):
        if index not in self.metainfo_cache:
            self.metainfo_cache[index] = self._get_metainfo(index)
        return self.metainfo_cache[index]

    def _get_metainfo(self, index):
        raise NotImplementedError()

class FilterableDataset(FilterableDatasetBase):
    def __init__(self, owner_dataset, indices = None, filter_name = None, filter_func = None, **kwargs):
        super().__init__()
        self.owner_dataset = owner_dataset
        self._filter_func = filter_func
        self._filter_name = filter_name
        self.indices = indices
        for key, value in kwargs.items(): 
            setattr(self,key, value)
       
            
    @property
    def filter_name(self):
        return self._filter_name if self._filter_name is not  None else "<anonumous>"
    
    @property
    def full_filter_name(self):
        if self.indices is not None:
            return self.owner_dataset.full_filter_name + '/' + self.filter_name
        return '<original>'

    @property
    def filter_func(self):
        return self._filter_func

    def collect(self, key_func):
        return {key_func(self.get_metainfo(i)) for i in range(len(self))}

    def filter(self, filter_func, filter_name=None):
        indices = []
        for i in range(len(self)):
            metainfo = self.get_metainfo(i)
            if filter_func(metainfo):
                indices.append(i)
        if len(indices) == 0:
            raise ValueError('Filter results in an empty dataset.')
        return type(self)(self, indices, filter_name, filter_func)

    def __getitem__(self, index):
        if self.indices is None:
            return self.owner_dataset[index]
        if self.owner_dataset is None:
            if index < self.split_pivot:
                return self.parent_datasets[0][index]
            return self.parent_datasets[1][index-self.split_pivot]
        return self.owner_dataset[self.indices[index]]

    def __len__(self):
        if self.indices is None:
            return len(self.owner_dataset)
        return len(self.indices)

    def get_metainfo(self, index):
        if self.indices is None:
            return self.owner_dataset.get_metainfo(index)
        return self.owner_dataset.get_metainfo(self.indices[index])
    
    def random_trim_length(self, length):
        assert length < len(self)
        logger.info('Randomly trim the dataset: #samples = {}.'.format(length))
        indices = list(random.sample(range(len(self)), length))
        return type(self)(self, indices=indices, filter_name='randomtrim[{}]'.format(length))

    def trim_length(self, length):
        assert length < len(self)
        logger.info('Trim the dataset: #samples = {}.'.format(length))
        return type(self)(self, indices=list(range(0, length)), filter_name='trim[{}]'.format(length))

    def split_trainval(self, split):
        assert split < len(self)
        nr_train = split
        nr_val = len(self) - nr_train
        logger.info('Split the dataset: #training samples = {}, #validation samples = {}.'.format(nr_train, nr_val))
        return (
                type(self)(self, indices=list(range(0, split)), filter_name='train'),
                type(self)(self, indices=list(range(split, len(self))), filter_name='val')
        )

    def split_kfold(self, k):
        assert len(self) % k == 0
        block = len(self) // k

        for i in range(k):
            yield (
                    type(self)(self, indices=list(range(0, i * block)) + list(range((i + 1) * block, len(self))), filter_name='fold{}[train]'.format(i + 1)),
                    type(self)(self, indices=list(range(i * block, (i + 1) * block)), filter_name='fold{}[val]'.format(i + 1))
            )

    def extend(self, dataset):
        if len(self.indices) == 0:
            raise ValueError("The dataset to be extended should not be empty")
        split_pivot = len(self.indices) 
        if dataset.indices is not None:
            assert type(dataset.indices) == list, "Indices of a filterable dataset should be list but got {}".format(type(dataset.indices))
            indices = [i for i in range(len(self.indices)+len(dataset.indices))]
        return type(self)(None, indices = indices, filter_name = "Union", filter_func = None, split_pivot = split_pivot, parent_datasets = (self,dataset)) 

    def __getattr__(self, __name: str):
        return getattr(self.owner_dataset, __name)
        