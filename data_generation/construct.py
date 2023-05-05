#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : construct.py
# Author : Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 20/05/2022
import sys

import os
import argparse
import configs
from manage import remove_sample
import pybullet as p
import pybullet_data as pd
import numpy as np
from panda.settings import camera_settings, ColorList
from panda.construct.program_generator import ProgramGenerator
import random

def construct_main(template_file, metadata_file, fps=240., width=1024, height=768, dataset_dir='dir', objects=None):
  """
  :Parameters:
  fps: float, Dataset Directory From Root'
  width: Width of GUI Window
  height: Height of GUI Window
  dataset_dir: Relative path to the Dataset Directory
  objects: list, Types of objects required in the scene
  template_file: str, template file for generating program and instructions
  metadata_file: str, metadata file for possible concepts and synonyms
  """


  #set parameters of GUI window
  timeStep = 1./fps

  if(not os.path.isdir(dataset_dir)):
    os.mkdir(dataset_dir)


  objects = configs.MainConfigs.get('required_object_types',['Cube','Lego'])

  object_counts = configs.MainConfigs.get('num_objects_per_type')
  
  if object_counts is None:
    while True:
      object_counts = {}
      for obj in objects:
        object_counts['num_' + obj.lower() + 's'] = 0
        
      for _ in range(configs.MainConfigs['MAX_OBJECTS']):
        obj = random.choice(objects)
        object_counts['num_' + obj.lower() + 's'] += 1

      for key in object_counts:
        if object_counts[key] > len(ColorList):
          continue
      break

  Generator = DataConstructor(timeStep,objects,object_counts)
  Generator.construct_data(height, width, dataset_dir, template_file, metadata_file)
  p.disconnect()


def init_bulletclient(timeStep, width=None, height=None, video_filename=None):
  #connection_mode GUI = graphical mode, DIRECT = non-graphical mode
  import pybullet as p
  import pybullet_data as pd
  if video_filename is None:
    p.connect(p.GUI)
  else:
    p.connect(p.GUI, options = f"--minGraphicsUpdateTimeMs=0 --width={width} --height={height} --mp4=\"{video_filename}\" --mp4fps=48")
  #Add pybullet_data path to search path to load urdf files
  p.setAdditionalSearchPath(pd.getDataPath())
  # p.setAdditionalSearchPath("./urdf/")
  

  #visualizer and additional settings 
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
  p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
  p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
  p.setPhysicsEngineParameter(solverResidualThreshold=0)
  # p.resetDebugVisualizerCamera(**camera_settings['small_table_view'])
  p.resetDebugVisualizerCamera(**camera_settings['big_table_view_2'])
  p.setTimeStep(timeStep)
  p.setGravity(0, 0, -10.0)
  p.setRealTimeSimulation(0)

class DataConstructor(object):
  def __init__(self,timeStep,objects,object_counts,max_objects=configs.MainConfigs['MAX_OBJECTS']):
    '''
    constructor: Options = ['NaiveSingleStep','NaiveDoubleStep','RelationalSingleStep','RelationalDoubleStep']
    objects(List) : Types of objects needed
    obj_count(List): No of each objects required in the scene. By default, the list is randomly generated.  
    '''
    init_bulletclient(timeStep)
    self.objects = objects
    self.max_objects = max_objects
    self.object_counts = object_counts
    self.config = self.generate_config()
    self.time_step = timeStep

  def generate_config(self):
    config = {}
    config['object_counts'] = self.object_counts
    config['rotation'] = configs.MainConfigs['rotation']
    config['num_instruction_per_scene'] = configs.MainConfigs['num_instruction_per_scene']
    if configs.MainConfigs['rotation'] == True:
        config['orn'] = p.getQuaternionFromEuler(configs.MainConfigs['euler_orientation']) if 'euler_orientation' in configs.MainConfigs.keys() else None
    for key in configs.AdditionalConfigs.keys():
        config[key] = configs.AdditionalConfigs[key]
    return config

  def construct_data(self, height, width, dataset_dir, template_file, metadata_file):
        
    
    construct = ProgramGenerator(p,[0,0,0],self.config,height,width, None ,template_file = template_file, metadata_file = metadata_file)
    construct.hide_panda_body()
    self.config['scene_data'] = construct.get_scene_info()
    p.disconnect()
   
    previous_programs = []
    for i in range(self.config.get('num_instruction_per_scene',1)):
      print("Instruction no : ",  i)
      #create new dir for the sample   
      sample_no = len(os.listdir(dataset_dir))
      smpl_dir = os.path.join(dataset_dir, "{0:0=4d}".format(sample_no))
      os.mkdir(smpl_dir)

      #init bullet and construct the world from the saved scene data
      init_bulletclient(self.time_step)
      construct = ProgramGenerator(p,[0,0,0],self.config,height,width, smpl_dir ,template_file = template_file, metadata_file = metadata_file)
      construct.hide_panda_body()
      
      construct.save_instance() #save the initial scene
      status = construct.generate_grounded_functional_program(object_choice = self.config.get('instantiation_type', 'random'), MAX_ATEMPTS = self.config.get('max_program_generation_atempts',100))
      if status == False:  # if no compatible program found, remove the sample 
        remove_sample(smpl_dir)
        return
      elif construct.get_program() in previous_programs:
            remove_sample(smpl_dir)
            continue
      construct.save_instance() #save the final scene
      program, command_lexed, command, language_complexity = construct.generate_instruction(complexity = self.config.get('complexity', None))
      construct.save_demonstration_info(command_lexed, command, language_complexity, program,  self.config)
      p.disconnect()
      # print(command, '\n', program)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fps', default=240., type=float, help='Dataset Directory From Root')
  parser.add_argument('--width', default=1024, help='Width of GUI Window')
  parser.add_argument('--height', default=768, help='Height of GUI Window')
  parser.add_argument('--root_dir', default=os.getcwd(), metavar='DIR', help='Root Directory')
  parser.add_argument('--dataset_dir', metavar='DIR', help='Relative path to the Dataset Directory')
  parser.add_argument("--objects", type = list, default = None, help='Types of objects required in the scene')
  parser.add_argument('--num_objects', type = dict, default = None, help = 'Num of objects for each requiredtype')
  parser.add_argument('--template_file', required = True, type = str, help = "template file for generating program and instructions")
  parser.add_argument('--metadata_file', type = str, required = True, help = "metadata file for possible concepts and synonyms")
  parser.add_argument('--type', type=str, required=True)
  parser.add_argument('--language', type=str, required=True)
  parser.add_argument('--max_objects', type=int, required=True)
  parser.add_argument('--num_examples', type=int, required=True)
  args = parser.parse_args()

  category = {
    'type': args.type,
    'num_objects': args.max_objects,
    'language': args.language
  }

  if category['type'] == 'any':
    required_object_types = ['Cube', 'Lego', 'Dice']
  elif category['type'] == 'cube':
    required_object_types = ['Cube']
        
  CommandLineConfigs = {
      'MAX_OBJECTS': category["num_objects"],
      'required_object_types': required_object_types,
      'rotation': False,
      'num_objects_per_type' : None
  }

  for key, value in CommandLineConfigs.items():
    configs.MainConfigs[key] = value
  if category['language'] == 'simple':
    configs.AdditionalConfigs['complexity'] = 'simple'
  else:
    configs.AdditionalConfigs['complexity'] = random.choice(['complex', 'compound'])


  for i in range(args.num_examples):
    try:
      construct_main(args.template_file, args.metadata_file, args.fps, args.width, args.height, args.dataset_dir, args.objects)
    except Exception as e:
      print(e)
