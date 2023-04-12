#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 19/06/2021

def is_unique(e, l: list) -> bool:
  return sum([e == el for el in l]) == 1

def get_smpl_dir(dataset_dir):
  import os
  sample_no = len(os.listdir(dataset_dir))
  smpl_dir = os.path.join(dataset_dir, "{0:0=4d}".format(sample_no))
  os.mkdir(smpl_dir)
  return smpl_dir
