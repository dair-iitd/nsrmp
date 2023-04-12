#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_analysis.py
# Author : Jiayuan Mao
# Modified by: Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 06/07/2022
#
#Acknowledgement: This file is adopted from NSCL-pytorch Release

"""
Program analytics tools.
"""

from collections import defaultdict
import six
import collections

from datasets.definition import gdef

__all__ = [
    'dfs_nsrmtree',
    'nsrmtree_contains_op', 'nsrmtree_get_depth', 'nsrmtree_stat_parameters',
    'nsrmtree_to_string', 'nsrmtree_to_string_full'
]


def dfs_nsrmtree(program):
    def dfs(pblock):
        yield pblock
        for i in pblock['inputs']:
            yield from dfs(i)

    return list(dfs(program))


def nsrmtree_contains_op(program, inspect_set):
    if isinstance(inspect_set, six.string_types):
        inspect_set = {inspect_set}
    inspect_set = set(inspect_set)
    for block in dfs_nsrmtree(program):
        if block['op'] in inspect_set:
            return True
    return False


def nsrmtree_get_depth(program):
    def dfs(pblock):
        if 'inputs' not in pblock or len(pblock['inputs']) == 0:
            return 1
        return 1 + max(dfs(p) for p in pblock['inputs'])
    try:
        return dfs(program)
    finally:
        del dfs


def nsrmtree_stat_parameters(program):
    result = collections.defaultdict(int)
    for pblock in dfs_nsrmtree(program):
        op = pblock['op']
        for x in gdef.all_signatures_dict[op][0]:
            if x in gdef.parameter_types:
                result[x] += 1
    return result


def nsrmtree_to_string(program):
    def dfs(pblock):
        ret = pblock['op'] + '('
        inputs = [dfs(i) for i in pblock['inputs']]
        ret += ','.join(inputs)
        ret += ')'
        return ret
    return dfs(program)


def nsrmtree_to_string_full(program):
    def dfs(pblock):
        ret = pblock['op'] + '('
        inputs = []
        for param_type in gdef.parameter_types:
            param_record = None
            if param_type in pblock:
                param_record = pblock[param_type]
            elif param_type + '_idx' in pblock:
                param_record = pblock[param_type + '_values'][pblock[param_type + '_idx']]

            if param_record is not None:
                param_str = '|'.join(param_record) if isinstance(param_record, (tuple, list)) else str(param_record)
                inputs.append(param_str)

        inputs.extend([dfs(i) for i in pblock['inputs']])
        ret += ','.join(inputs)
        ret += ')'
        return ret

    return dfs(program)

def concepts_in_nsrmtree(program):
    concept_dict = defaultdict(list)
    def dfs(block):
        if block['op'] =='move':
            dfs(block['inputs'][-1])
            for pblock in block['inputs'][0:2]:
                dfs(pblock)
        else:
            for pblock in block['inputs']:
                dfs(pblock)
        for type in gdef.parameter_types:
            if type+'_idx' in block:
                concept_dict[type].append(block[type+'_idx'])
            
    dfs(program)
    return concept_dict
