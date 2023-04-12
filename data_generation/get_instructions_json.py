#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_engine.py
# Authors : Vishl Bindal, Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 20/05/2022

import json
import argparse
import os
from datetime import date
import numpy as np

def get_instruction_info(dir, id, split):
    with open(os.path.join(dir, 'demo.json'), 'r') as f:
        demo_json = json.load(f)

    example_dir = dir.split('/')[-1]

    instruction = {
        'ex_dir': example_dir,
        'id': id,
        'template_id': demo_json['template_id'],
        'template_json_filename':demo_json['template_json_filename'].split('/')[-1],
        'program': demo_json['program'],
        'grounded_program': demo_json['grounded_program'],
        'language_complexity': demo_json['language_complexity'],
        'instruction': demo_json['instruction'],
        'instruction_lexed': demo_json['instruction_lexed'],
        'split':split,
    }

    return instruction

def main(dataset_dir, split, out_path, out_dir="."):
    dump = {
        'info':{
            'date': str(date.today()),
            'split': split
        },
        'instructions': []
    }

    occlusion_path = os.path.join(out_dir, f'occlusion-cases-{split}.txt')
    occlusion_dirs = ""
    if os.path.exists(occlusion_path):
        f = open(occlusion_path, 'r')
        occlusion_dirs = "\n".join(f.readlines())


    all_dirs = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    all_dirs.sort()
    for id, dir in enumerate(all_dirs):
        if dir in occlusion_dirs:
            continue
        instruction = get_instruction_info(dir, id, split)
        dump['instructions'].append(instruction)

    with open(os.path.join(out_dir, out_path), 'w') as f:
        json.dump(dump, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required = True, type = str, help='Relative path to the Dataset Directory')
    parser.add_argument('--split', required = True, type = str,  help='Split (train/val/test)')
    parser.add_argument('--out_path', default='instructions.json', help='Output path')
    args = parser.parse_args()

    main(args.dataset_dir, args.split, args.out_path)