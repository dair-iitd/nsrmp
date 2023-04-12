#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   program_transforms.py
#Time    :   2022/06/16 23:28:31
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

#Acknowledgement: This  code is  the modified version of the translator used bt jiayuan mao in NSCL- Pytorch Release

from copy import deepcopy

def get_block_function(block):
    if "type" in block:
        return block['type']
    else:
        raise NotImplementedError


def get_roboclevr_op_attribute(op):
    return op.split('_')[1]


def roboclevr_to_nsrm(prog):
    program = deepcopy(prog)
    assert type(program) == list and type(program[0]) == dict, 'The program has to be a list of dictionary'
    nsrm_program = list()
    input_mapping = dict()

    for block_id, block in enumerate(program):
        current = None
        block_function = get_block_function(block)

        #Set operation name
        if block_function == 'scene':
            current = dict()
            current['op'] = 'scene'

        elif block_function.startswith('filter'):
            concept = block['value_inputs'][0].lower()
            current = dict(op='filter', attribute_concept=[concept], param_type = 'attribute_concept')
            
        elif block_function in ('intersect', 'union'):
            current = dict(op=block_function)

        elif block_function == 'relate':
            concept = block['value_inputs'][0].lower()
            current = dict(op = 'relate', relational_concept =[concept], param_type = 'relational_concept')

        elif block_function == 'unique':
            pass
        elif block_function in ['idle', 'move']:
            if block_function == 'idle':
                current = dict(op = 'idle')
            elif block_function == 'move':
                concept = block['value_inputs'][0].lower()
                current = dict(op = 'move', action_concept = [concept], param_type = 'action_concept')
        else:
            raise ValueError('Unknown block operation')
        
        #Set the action bool variable
        if current is not None:
            current['action'] = block['action']
        
        
        #set the inputs
        if current is None:
            assert len(block['inputs']) == 1 
            input_mapping[block_id] = input_mapping[block['inputs'][0]]
        else:
            current['inputs'] = list(map(input_mapping.get,block['inputs']))

            #set the outputs
            if 'output' in block:
                current['output'] = ['world'] if block['output'] is None else block['output']
            nsrm_program.append(current)
            input_mapping[block_id] = len(nsrm_program) - 1
    return nsrm_program





from copy import deepcopy     
def append_program_tree_pairs(program_1,program_2,cgs,rcgs,ags):
    #both program_1 and program_2 are imperatives is assumed and the inputs in the list are ordered with the incoming worl
    #d state as the last argument 
    combined_cg,combined_rcg,combined_ag=  cgs[0]+cgs[1],rcgs[0]+rcgs[1],ags[0]+ags[1]
    if program_1['op'] == 'idle':
        return program_2,cgs[1],rcgs[1],ags[1]
    
    if program_2['op'] == 'idle':
        return program_1,cgs[0],rcgs[0],ags[0]
    

    combined_program = deepcopy(program_2)

    def dfs(program,change_idx):
        
        if program.get('param_type',None):
            if program['param_type'] == 'action_concept':
                if change_idx: 
                    program['action_concept_idx'] += len(ags[0])
                program['action_concept_values'] = combined_ag 

            elif program['param_type'] == 'relational_concept':
                if change_idx: 
                    program['relational_concept_idx'] += len(rcgs[0])
                program['relational_concept_values'] = combined_rcg 

            elif program['param_type'] == 'attribute_concept':
                if change_idx:
                    program['attribute_concept_idx'] += len(cgs[0])
                program['attribute_concept_values'] = combined_cg 
            
            for sub_program in program['inputs']:
                dfs(sub_program, change_idx)
    
    dfs(combined_program, True)
    append_program = deepcopy(program_1)
    dfs(append_program,change_idx=False)
    combined_program['inputs'][-1] = append_program 
    
    return combined_program,combined_cg,combined_rcg,combined_ag

def append_program_trees(program_tree_list, cgs,rcgs,ags):

   
    combined_program,combined_cgs,combined_rcgs,combined_ags = program_tree_list[0],cgs[0],rcgs[0],ags[0]

    for i in range(1,len(program_tree_list)):
        combined_program,combined_cgs,combined_rcgs,combined_ags = append_program_tree_pairs(combined_program,program_tree_list[i],
        [combined_cgs,cgs[i]], [combined_rcgs,rcgs[i]], [combined_ags,ags[i]])


    return combined_program,combined_cgs,combined_rcgs,combined_ags