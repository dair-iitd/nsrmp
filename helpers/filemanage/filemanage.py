#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   filemanage.py
#Time    :   2022/06/10 21:05:54
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in



import os

def ensure_path(path):
    if os.path.exists(path):
        return
    print("creating a directory: '{}'".format(path))
    os.makedirs(path,exist_ok=True)