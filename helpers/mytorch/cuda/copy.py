#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   copy.py
#Time    :   2022/07/01 15:14:30
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

import collections
import torch

#Adapted from Jacinle by Jiayuaon Mao
#https://github.com/vacancy/Jacinle/blob/master/jactorch/cuda/copy.py

def async_copy_to(obj, dev, main_stream=None):
    """
    Copy an object to a specific device asynchronizedly. If the param `main_stream` is provided,
    the copy stream will be synchronized with the main one.

    Args:
        obj (Iterable[Tensor] or Tensor): a structure (e.g., a list or a dict) containing pytorch tensors.
        dev (int): the target device.
        main_stream (stream): the main stream to be synchronized.

    Returns:
        a deep copy of the data structure, with each tensor copied to the device.

    """
    # Adapted from: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/_functions.py
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj
