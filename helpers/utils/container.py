#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   container.py
#Time    :   2022/06/08 21:04:01
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in


import copy
import collections

__all__ = ["DOView"]

class DOView(dict):
    def __setattr__(self, key, value):
        self[key] = value
    
    def __getattr__(self,key):
        if key not in self:
            raise AttributeError
        else:
            return self[key]
    
    def __delattr__(self, key) -> None:
        del self[key]

    def make_dict(self):
        info = {}
        for key, value in self.items():
            info[key] = value
        return info