#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : program_generator.py
# Author : Namasivayam K
# Email  : namasivayam.k@cse.iitd.ac.in
# Date   : 30/05/2022


import json
import random
import re
from copy import deepcopy
from .scene_graph import SceneGraph
from .program_engine import ProgramExecutor


class ProgramGenerator(ProgramExecutor):
    def __init__(self, bullet_client, offset, config, height, width, instance_dir,template_file,metadata_file= './metadata.json'):
        self.template_file = template_file
        self.template = self.load_template(template_file)
        self.metadata = self.load_metadata(metadata_file)
        self.config = config
        super().__init__(bullet_client, offset, config, height, width, instance_dir)
        self.save_position_info()

    def load_template(self,template_file):
        with open(template_file,'r') as f:
            all_templates = json.load(f)
        return random.choice(all_templates)
    
    def load_metadata(self, metadata_file):
        with open(metadata_file,'r') as f:
            metadata = json.load(f)
        return metadata


    def get_substitute(self,char_token, index):
        #Note: The index starts from 1 in the template. So reduce the indices by -1 in the subsequent part.
        if char_token == 'T':
            return self.objects[self.unique_objects[index -1 ]].type
        elif char_token == 'C':
            return self.objects[self.unique_objects[index -1 ]].color[1]
        elif char_token == 'A':
            return self.unique_actions[index-1]
        elif char_token == 'R':
            return self.relations[index -1]
        else:
            raise ValueError("Unknown char token {}".format(char_token))



    def generate_instruction(self,complexity=None):
        '''
        Description: The instructions are read from the template and the <A>, <C>, <O> tokens are replaced with the appropriate concept words.
        Inputs: complexity(str): options = ['simple', 'complex', 'compound']. Please pass this in the config dict. If None, one of the options is randomly choosen.

        Output: program:(list of 3-tuples)
                instruction:(string)
        '''
        complexity = random.choice(['simple','complex','compound']) if complexity is None else complexity
        sent_lexed = random.choice(self.template['text'][complexity])
        words = sent_lexed.split()
        
        for idx,w in enumerate(words):
            match = re.search("<(\w)(\d)>",w)
            if match:
                char_token = match.group(1)
                if char_token == "T":
                    substitute = self.get_substitute('T',int(match.group(2)))
                    substitute = substitute.lower()
                    if substitute == 'lego':
                        substitute += ' block'
                    words[idx] = substitute 
                elif char_token == 'C':
                    substitute = self.get_substitute('C', int(match.group(2)))
                    substitute = substitute.lower()
                    words[idx] = substitute 
                elif char_token == 'A':
                    substitute = self.get_substitute('A', int(match.group(2)))
                    substitute = substitute.lower()
                    words[idx] = substitute+'_a'
                elif char_token == 'R':
                    substitute = self.get_substitute('R', int(match.group(2)))
                    substitute = substitute.lower()
                    words[idx] = substitute
                else:
                    raise ValueError("Unknown char token {}".format(char_token))
        #At this point, the instruction will be created. But we replace some of the words with thier synonyms from the metadata. 
        sent = ' '.join(words)
        words = sent.split()
        synon = self.metadata['synonyms']
        for idx,w in enumerate(words):
            w = w.lower()
            if w in synon.keys():
                replace_text = random.choice(synon[w])
                words[idx] = replace_text
        return self.get_program(), sent_lexed, ' '.join(words), complexity


    def generate_grounded_functional_program(self, object_choice = 'default', MAX_ATEMPTS = 10000):
        '''
         Inputs:  object_choice(string): options= ['default','random']. By default, the first n objects are used. 

        '''
        template = self.template
        is_relational = template.get("relational", False)
        num_objects = sum(list(self.config['object_counts'].values()))
        try:
            assert num_objects >= template['num_unique_objects']
        except AssertionError:
            print( " The number of  unique objects to be intialized is less than the no of objects in the world")
            return False
        for i in range(MAX_ATEMPTS):
            #choose actions and objects instances
            self.unique_actions = random.sample(self.metadata['actions']['move'],template['num_unique_actions']) 
            self.unique_objects = [i for i in range(template['num_unique_objects'])] if object_choice == 'default' else random.sample([i for i in range(num_objects)],template['num_unique_objects'])
            if is_relational:
                all_relations = self.config.get('relations', self.metadata['relations']['spatial_relations'])
                self.relations = [random.choice(all_relations) for i in range(template['num_relations'])]
            self.symbolic_program = deepcopy(self.template['nodes'])
            self.scene_graph = SceneGraph(self.objects,self.position_list[0],self.config)
            self.program = list()
            for node in self.symbolic_program:
                # Replace the tokens by the respective concept words
                for idx,str in enumerate(node['value_inputs']):
                    match = re.search("<(\w)(\d)>", str)
                    if match:
                        node['value_inputs'][idx] = self.get_substitute(match.group(1),int(match.group(2)))
                    else:
                        breakpoint()
                        raise NotImplementedError
            
            # Try Executing the program on the scene
            status = self.execute_symbolic_program()
            if status == True:
                print("##### Found one program ###### ")
                print("\n...........Requesting PyBullet to perform the Simulation...............")
                self.target_position = self.check_action_compatibility(self.get_program(),self.position_list[-1])
                assert self.target_position is not None , "There is a bug in program generation"
                self.apply_program()
                return True
            else:
                self.program = []

        if len(self.program) == 0:
            print("*******NO compatible program found *******")
            return False
 
    def get_program(self):
        return self.program

    

    def save_demonstration_info(self, command_lexed, command, complexity, program,  configs):
        info = dict()
        info['instruction'] = command
        info['instruction_lexed'] = command_lexed
        scene_info = self.get_scene_info()
        info.update(scene_info)
        info['template_json_filename'] = self.template_file
        info['template_id'] = self.template['template_id']
        info['language_complexity'] = complexity
        info['is_question'] = False
        info['program'] = self.symbolic_program
        info['grounded_program'] = program
        with open(f"{self.instance_dir}/demo.json", 'w') as write_file:
            json.dump(info, write_file)

    def save_question_info(self, question):
        info = dict()
        info['question'] = question
        scene_info = self.get_scene_info()
        info.update(scene_info)
        info['template_filename'] = self.template_file
        info['is_question'] = True
        info['program'] = self.symbolic_program
        with open(f"{self.instance_dir}/question.json", 'w') as write_file:
            json.dump(info, write_file)

