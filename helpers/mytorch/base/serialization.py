#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   serialization.py
#Time    :   2022/08/23 16:24:48
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in

import torch
import torch.nn as nn
import sys
from helpers.logging import get_logger

logger = get_logger(__file__)


def _belongs(key, module_list):
    for k in module_list:
        if k in key:
            return True
    return False

def load_state_dict(model:nn.Module, saved_model_path, partial = False, **kwargs):
    try:
        if not partial:
                logger.critical(f'Trying to load the complete model from {saved_model_path}')
                model.load_state_dict(torch.load(saved_model_path))
        else:
            saved_model_state_dict = torch.load(saved_model_path)
            model_dict = model.state_dict()
            modules_to_be_loaded = kwargs.get('modules', None)
            if modules_to_be_loaded is not None:
                assert type(modules_to_be_loaded) == list
                logger.critical(f'Trying to load {modules_to_be_loaded} from {saved_model_path}')
                pretrained_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_dict and _belongs(k,modules_to_be_loaded)}
            else:
                logger.critical(f'Trying to load all matching keys from {saved_model_path}')
                pretrained_dict = {k: v for k, v in saved_model_state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        logger.info('Loaded the model succesfully!\n')
    except Exception as e:
        logger.error(f'\nError in loading model', exc_info = True, stack_info = True)
        sys.exit("Some error occured in loading the  model. Please check if the model being loaded is compatible with the current architecture ")