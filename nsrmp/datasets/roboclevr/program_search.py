#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_search.py
#Modifier : Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 06/07/2022
# 
# Acknowledgemet: This file was adopted and modified from Jiayuan Mao NSCL-Pytorch release

import itertools
import os.path as osp
from copy import deepcopy
# from functools import lru_cache
from cached_property import cached_property
import helpers.io as io
from helpers.utils.container import DOView
from datasets.definition import gdef
from datasets.common.program_translator import nsrmtree_to_nsrmseq
from datasets.common.program_analysis import nsrmtree_stat_parameters, dfs_nsrmtree

# @lru_cache(maxsize=1)
# def load_program_templates(filename):
#     templates = io.load_json(osp.join(osp.dirname(__file__), filename))
#     ret = dict()
#     for key, value in templates.items():
#         stat = nsrmtree_stat_parameters(value)
#         stat = tuple(stat.get(param_type, 0) for param_type in gdef.parameter_types)
#         ret[key] = (value, stat)
#     return ret



class SearchCandidatePrograms(object):
    def __init__(self, group_concepts, transform, template_filename):
        self.group_concepts = group_concepts
        self.template_filename = template_filename
        self.transform = transform

    def __call__(self, *args):
        tokens, concept_dict = self.transform(*args, group_concepts = self.group_concepts)
        programs = self.get_candidate_programs(concept_dict)
        out_dict = {'program_parser_candidates_qstree':programs}
        out_dict.update(concept_dict)
        return tokens, out_dict 

    @cached_property
    def load_program_templates(self):
        templates = io.load_json(osp.join(osp.dirname(__file__), self.template_filename))
        ret = dict()
        for key, value in templates.items():
            stat = nsrmtree_stat_parameters(value)
            stat = tuple(stat.get(param_type, 0) for param_type in gdef.parameter_types)
            ret[key] = (value, stat)
        return ret
    

    def get_candidate_programs(self, concept_dict):
        q_stat = tuple(len(concept_dict.get(param_type + 's',[])) for param_type in gdef.parameter_types)
        candidate_meta_programs = list()
        for summary, (prog, prog_stat) in self.load_program_templates.items():
            if q_stat == prog_stat:
                candidate_meta_programs.append((summary, prog))
        candidate_programs = dict()
        for summary, metaprog in candidate_meta_programs:
            metaprog = deepcopy(metaprog)
            for pblock in dfs_nsrmtree(metaprog):
                for param_type in gdef.parameter_types:
                    if param_type + '_values' in pblock:
                        pblock[param_type + '_values'] = concept_dict[param_type + 's']

            candidate_programs[summary] = list()

            permutations = [
                itertools.permutations(range(len(concept_dict.get(param_type + 's',[]))))
                for param_type in gdef.parameter_types
            ]
            for choice_tuple in itertools.product(*permutations):
                choice = dict(zip(gdef.parameter_types, choice_tuple))
                counter = {pt: 0 for pt in gdef.parameter_types}
                prog = deepcopy(metaprog)
                for pblock in dfs_nsrmtree(prog):
                    for param_type in gdef.parameter_types:
                        if param_type + '_idx' in pblock:
                            cnt = counter[param_type]
                            counter[param_type] += 1
                            val = choice[param_type][cnt]
                            pblock[param_type + '_idx'] = val
                            pblock[param_type] = pblock[param_type + '_values'][pblock[param_type + '_idx']]

                candidate_programs[summary].append(prog)
        return candidate_programs

