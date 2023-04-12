#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   definition.py
#Time    :   2022/06/11 16:57:53
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

#Acknowledgement: Few parts of the code is adopted from jiayuan Moa - NSCL-PyTorch Release
from cached_property import cached_property
from copy import deepcopy

__all__ = ['DefinitionBase', 'gdef', 'get_global_defn','set_global_defn']

class DefinitionBase(object):

    def get_concepts_by_type(self,concept_type):
        
        assert concept_type in gdef.concept_groups, "Invalid concept type"
        concept_container = getattr(self,concept_type)
        all_concepts = []
        if type(concept_container) == list:
            all_concepts.extend(concept_container)
        elif type(concept_container) == dict:
            for v in concept_container.values():
                all_concepts.extend(v)
        else:
            raise NotImplementedError('Unknown container type')
        return all_concepts

    @cached_property
    def all_concept_words(self):
        all_concept_words = []
        for c in gdef.concept_groups:
            all_concept_words.extend(self.get_concepts_by_type(c))
        return all_concept_words
        
    @cached_property
    def concept_words_dict(self):
        return {k:gdef.get_concepts_by_type(k) for k in gdef.concept_groups}

    def require_concept(self, operation):
        return len(self.all_signatures_dict[operation][0]) > 0

    @property
    def all_types(self):
        return self.parameter_types + self.variable_types + self.return_types
    
    @cached_property
    def all_signatures_dict(self):
        _sig_dict = deepcopy(self.operation_signatures_dict)
        if getattr(self,'action_signatures_dict', None) is not None:
            _sig_dict.update(self.action_signatures_dict)
        return _sig_dict

    @cached_property
    def operation_signatures_dict(self):
        return {v[0]: v[1:] for v in self.operation_signatures}

    # Automatically generated type mappings.
    @property
    def qtype2atype(self):
        return [
            (name, ret_type) for name, _, _, ret_type in self.operation_signatures  # if ret_type in self.return_types
        ]

    @property
    def qtype2atype_dict(self):
        return dict(self.qtype2atype)

    @property
    def atype2qtypes(self):
        atype2qtypes = dict()
        for k, v in self.qtype2atype:
            atype2qtypes.setdefault(k, []).append(v)
        return atype2qtypes
    
class GlobalDefinitionFinder(object):
    def __getattr__(self, attr):
        return getattr(get_global_defn(), attr)


_GLOBAL_DEF = None

def get_global_defn():
    return _GLOBAL_DEF   

def set_global_defn(defn):
    global _GLOBAL_DEF
    if _GLOBAL_DEF is not None:
        raise ValueError("Global defn is already set")
    _GLOBAL_DEF = defn

gdef = GlobalDefinitionFinder()