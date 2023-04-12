#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 30/05/2022

import numpy as np
import random
from panda.settings import object_dimensions
from panda.settings import directions as setting_direction

class SceneGraph(object):
    def __init__(self,objects,positions, config):
        self.objects = [o.__dict__ for o in objects]
        self.positions = positions
        self.directions = config.get('directions',setting_direction)
        self.eps = config.get('relation_epsilon', 0.075)
        self.relationships = self.find_relations()

    def find_relations(self):
        num_obj = len(self.objects)
        relationships = {}
        for dir, vec in self.directions.items():
            relationships[dir] = []
            for i in range(num_obj):
                coords_i = self.positions[i]
                related_i = []
                for j in range(num_obj):
                    if j==i:
                        continue
                    coords_j = self.positions[j]
                    diff = np.array(coords_j) - np.array(coords_i)
                    dot = np.dot(np.array(vec), diff)
                    if dot > self.eps:
                        related_i.append(j)
                relationships[dir].append(related_i)
        return relationships
        
    def update(self,positions):
        self.positions = positions
        self.relationships = self.find_relations()
