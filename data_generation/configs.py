#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : configs.py
# Author : Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 20/05/2022



'''
required_object_types(List): The objects that we want in the scene - a sublist of ['Cube','Tray','Lego','Dice']
num_objexts_per_type(dict): keys = [num_+Obj[0].lower()+Obj[1:]+'s' for Obj in required_object_types]. If not given, the num of objects for each required_type  will be randomly choosen
rotation(Bool): If True the objects will be rotated randomly which the scene gets initialized
euler_orientation(3-tuple): By default orientation is same as that of the object in the urdf file. No rotation is applied. Provide a value if you want all the objects to be rotated by a same value.

'''


MainConfigs = {
'MAX_OBJECTS': 5,
'required_object_types' :['Cube'], #Type: List, Options: 'Cube', 'Lego', 'Dice', Default: ['Cube', 'Lego']
'num_objects_per_type' :{'num_cubes':3,'num_legos':2,'num_dices':1}, #If not given, num_objects will get randomly initialized
'rotation': False,
'num_instruction_per_scene':3,
#euler_orientation = (0,0,0) 
}

AdditionalConfigs = {
   'instantiation_type' : 'random', #Type: str, options: ['random', 'default'], Default: 'default
   'complexity' : 'simple', #options: ['simple', 'complex', 'compound'], Default: Randomly choosen
   'relations' : ['left', 'right', 'behind', 'front'], #Use this to restrict relations. If this key is not there, the relational_concepts will be sampled from ["left", "right", "behind", "front"]
   'max_program_generation_atempts': 3000 #For each scene, the simulator will try to find a compatible program. This key restricts the number of such attempts. If all attempts failed, then the scene will get deleted.
}

#future additional configs :  currently in development
# 'directions' = {} 