#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_translator.py
# Author : Jiayuan Mao
# Modifier : Namasivayam K
# Date   : 17/06/2021
#
# This file was part of NSCL-PyTorch and was adopted and modified by the authors of NSRMP
# Distributed under terms of the MIT license.


from copy import deepcopy
from collections  import defaultdict
from datasets.definition import gdef

def nsrmseq_to_nsrmtree(seq_program):
    def dfs(sblock):
        tblock = deepcopy(sblock)
        input_ids = tblock.pop('inputs')
        tblock['inputs'] = [dfs(seq_program[i]) for i in input_ids]
        return tblock

    try:
        return dfs(seq_program[-1])
    finally:
        del dfs


def nsrmtree_to_nsrmseq(tree_program):
    tree_program = deepcopy(tree_program)
    seq_program = list()

    def dfs(tblock):
        sblock = tblock.copy()
        input_blocks = sblock.pop('inputs')
        sblock['inputs'] = [dfs(b) for b in input_blocks]
        seq_program.append(sblock)
        return len(seq_program) - 1

    try:
        dfs(tree_program)
        return seq_program
    finally:
        del dfs


def nsrmseq_to_nsrmqsseq(seq_program):
    qs_seq = deepcopy(seq_program)
    cached = defaultdict(list)

    for sblock in qs_seq:
        for param_type in gdef.parameter_types:
            if param_type in sblock:
                sblock[param_type + '_idx'] = len(cached[param_type])
                sblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(sblock[param_type])

    return qs_seq


def nsrmtree_to_nsrmqstree(tree_program):
    qs_tree = deepcopy(tree_program)
    cached = defaultdict(list)

    for tblock in iter_nsrmtree(qs_tree):
        for param_type in gdef.parameter_types:
            if param_type in tblock:
                tblock[param_type + '_idx'] = len(cached[param_type])
                tblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(tblock[param_type])

    return qs_tree


def iter_nsrmtree(tree_program):
    yield tree_program
    for i in tree_program['inputs']:
        yield from iter_nsrmtree(i)

#append tree1 to bottom of tree2
def append_twotrees(tree1,tree2):
    if len(tree2) == 0:
        return tree1
    tree = deepcopy(tree2)
    def dfs(tree):
        if tree['inputs'][-1]['op'] != 'idle':
            return dfs(tree['inputs'][-1])
        return tree
        
    if tree['op'] !='idle':
        last_node = dfs(tree)
        last_node['inputs'][-1] = deepcopy(tree1)
    else:
        if len(tree1)!=0:
            return tree1
    return tree

def append_nsrmtrees(trees):
    combined_tree = trees[0]
    for i in range(1, len(trees)):
        combined_tree = append_twotrees(combined_tree,trees[i])
    return combined_tree

def break_nsrmtrees(trees):
    if type(trees) == dict:
        trees = [trees]
    assert type(trees) == list  
    output_trees = []
    for tree in trees:
        this_forest = []
        def dfs(program,forest:list):
            if program['inputs'][-1]['op'] != 'idle':
                dfs(program['inputs'][-1], forest)
                program['inputs'][-1] = {'op':'idle', 'inputs':[]}
            forest.append(program)
            
        dfs(deepcopy(tree), this_forest)
        output_trees.append(this_forest)
    return output_trees




