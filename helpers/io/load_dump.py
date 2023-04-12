#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   load_dump.py
#Time    :   2022/06/10 21:06:23
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in



import json

def load_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data





def dump_json(filename,data):
    with open(filename,'w') as f:
        json.dump(data, f)
    