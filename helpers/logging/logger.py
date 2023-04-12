#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   logger.py
#Time    :   2022/06/10 21:06:45
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in


import logging
import os.path as osp
from helpers.filemanage import ensure_path

_default_level_ = logging.DEBUG

def get_logger(name = None):
    logger = logging.getLogger(name)
    logger.setLevel(_default_level_)
    return logger


def set_log_output_file(dir, name):
    ensure_path(dir)
    logging.basicConfig(filename = osp.join(dir,name), format = '%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m %H:%M:%S')