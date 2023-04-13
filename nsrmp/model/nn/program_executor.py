#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   program_executor.py
#Time    :   2022/06/26 14:03:33
#Authors  :   Namasivayam K, Himanshu G Singh
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release


import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.definition import gdef

from .concept_embedding import ConceptEmbeddings
from .action_simulator import ActionSimulator

from helpers.logging import get_logger

logger = get_logger(__file__)

class ProgramExecutor(nn.Module):
    def __init__(self, visual_feature_dim, emb_dim, nr_actions, used_concepts_dict = None):
        super().__init__()
        self.concept_embeddings = nn.ModuleDict()
        self.action_sim = ActionSimulator(nr_actions = nr_actions )
        if used_concepts_dict is not None:
            raise NotImplementedError
        else:
            logger.info("Initializing Concept Embeddings Using global_dataset_definition gdef")
            all_concept_groups = gdef.concept_groups
            for type in all_concept_groups:
                if type in emb_dim:
                    self.concept_embeddings['embedding_'+type] =  ConceptEmbeddings(attribute_agnostic=False)
                    self.concept_embeddings['embedding_'+type].init_from_gdef(type, visual_feature_dim, emb_dim[type])
                

    def forward(self, programs, visual_features, initial_bboxes, **kwargs):
        #@Namas: TODO - currently kwargs is used to set the parameters for unique mode. Rewrite this in a elegant way when time permits
 
        grounded_programs, buffers, results = [], [], []
        movements = []

        for i,prog_info in enumerate(programs):
            buffer = []
            g_program = []
            buffers.append(buffer)
            cur_scene_idx = prog_info['scene_id'] 
            prog = prog_info['program']
            log_likelihood = prog_info['log_likelihood']
            bboxes = initial_bboxes[cur_scene_idx]
            self.features = visual_features[cur_scene_idx]
            assert self.features[0].size(0) == bboxes.size(0)
            num_objects = bboxes.size(0)
            

    
            for block in prog:
                op = block['op']

                inputs = []
                #get inputs
                for inp_idx, inp_type in zip(block['inputs'],gdef.all_signatures_dict[op][1]):
                    inp = buffer[inp_idx]
                    if inp_type == 'object':
                        inp = self.unique(inp,**kwargs)
                    inputs.append(inp)
               
                #get concepts
                concepts = None
                if gdef.require_concept(op):
                    param_type = block.get('param_type', None)
                    concepts = block.get(param_type + '_values')[block.get(param_type+'_idx')]
        
                #execute block
                if op in gdef.actions:
                    # assert block['action'] == True
                    if op == 'idle':
                        buffer.append(bboxes)
                    elif op == 'move':
                        if 'gt_programs' in kwargs:
                            m_obj_idx, b_obj_idx = torch.zeros(num_objects, device = initial_bboxes[0].device), torch.zeros(num_objects, device = initial_bboxes[0].device)
                            m_idx, b_idx = kwargs['gt_programs'][i][0][1:]
                            m_obj_idx[m_idx] = 1
                            b_obj_idx[b_idx] = 1
                            bboxes_i = inputs[-1]
                        else:
                            m_obj_idx, b_obj_idx, bboxes_i = inputs

                        g_program.append((concepts[0], m_obj_idx, b_obj_idx))
                        bbox_1,bbox_2 = torch.matmul(b_obj_idx,bboxes_i), torch.matmul(m_obj_idx,bboxes_i)
                        action = torch.tensor(self.concepts_to_indices(concepts,param_type), device = bbox_1.device).to(torch.float)
                        # action = torch.tensor([[block['relational_concept'] == x for x in gdef.relational_concepts]]).view(-1)
                        bbox_pred = self.action_sim(bbox_1,bbox_2,action)
                        movements.append((bbox_2, m_obj_idx, bbox_pred))
                        bboxes_f = torch.mul(bboxes_i,(1-m_obj_idx).view(-1,1)) + torch.mul(bbox_pred.repeat(num_objects,1),m_obj_idx.view(-1,1))
                        buffer.append(bboxes_f)
                else:
                    if op == 'scene':
                        buffer.append(torch.ones(num_objects, device = initial_bboxes[0].device)*10)
                    elif op == 'filter':
                        buffer.append(self.filter(*inputs,concepts))
                    elif op == 'relate':
                        buffer.append(self.relate(*inputs, concepts))

            results.append(dict(scene_id = cur_scene_idx, pred_bboxes =  buffer[-1], log_likelihood = log_likelihood, movements = movements))
            grounded_programs.append(dict(scene_id = cur_scene_idx, grounded_program = g_program, log_likelihood = log_likelihood))

        return programs, grounded_programs, results, buffers            
                
    @staticmethod
    def concepts_to_indices(concepts,param_type):
        # all_concepts = gdef.get_concepts_by_type(gdef.param_to_concept[param_type])
        all_concepts = ['left','right','top']
        out = [0]*len(all_concepts)
        for i,c in enumerate(all_concepts):
            if c in concepts:
                out[i] = 1 
        return out

                
    def unique(self,inputs,**kwargs):
        unique_mode = kwargs.get('unique_mode','softmax')
        tau = kwargs.get('gumbel_tau', 0.3)

        if unique_mode == 'softmax':
            return F.softmax(inputs,dim=-1)
        elif unique_mode == 'gumbel':
            return F.gumbel_softmax(inputs, dim =-1, tau = tau, hard = True)
        elif unique_mode == 'argmax':
            out = torch.zeros(inputs.size(0), device = inputs.device)
            max_idx = inputs.argmax()
            out[max_idx] = 1
            return out
        else:
            raise NotImplementedError

    _margin = 0.85
    _tau = 0.25

    def filter(self,mask, concepts):
        if len(concepts) >1: 
            return self.filter_recursive(mask, concepts)

        new_mask = self.similarity(self.features[0], concepts[0], 'attribute_concepts')
        return torch.min(new_mask, mask) 
    
    def similarity(self, features, concept_identifier, embedding_identifier):
        concept = self.concept_embeddings['embedding_'+embedding_identifier].get_concept(concept_identifier)
        ops = self.concept_embeddings['embedding_'+embedding_identifier].get_all_attribute_operators
        f_mapped = torch.stack([m(features) for m in ops], dim=-2)
        f_mapped = f_mapped/f_mapped.norm(2,dim=-1,keepdim=True)
        
        logits_all_attributes = (torch.mul(f_mapped, concept.normalized_embedding).sum(dim=-1) -1 + self._margin)/self._margin/self._tau
        logits_marginialized =  torch.mv(logits_all_attributes,concept.normalized_belong)
        return logits_marginialized
       
       
    def filter_recursive(self,mask, concepts):
        old_mask = mask
        for c in concepts:
            new_mask = self.similarity(self.features[0],c, 'attribute_concepts')
            old_mask = torch.min(old_mask,new_mask)
        return old_mask
    
    def relate(self, obj_id, rel_concept):
        obj_id = torch.argmax(obj_id)
        try:
            new_mask = self.similarity(self.features[1][obj_id], rel_concept[0], 'relational_concepts')
        except :
            print("Fail in relate execution")
        return new_mask

