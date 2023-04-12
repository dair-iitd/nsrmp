#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   model_new.py
#Time    :   2022/06/27 20:36:07
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release


from shutil import move
from click import command
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from helpers.utils.container import DOView
from datasets.definition import gdef

from helpers.mytorch.vision.ops.boxes import box_convert, normalize_bbox

class Identity(nn.Module):
    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return inputs

class Model(nn.Module):
    def __init__(self, vocab, configs, training_target = 'all', resnet_pretrained=True):
        super().__init__()
        self.training_target = training_target
        self.configs  = configs
        from model.nn.visual_new import ObjectEmbedding
        import jactorch.models.vision.resnet as resnet 
        self.resnet = resnet.resnet34(pretrained=resnet_pretrained, incl_gap= False, num_classes = None)
        self.resnet.layer4 = Identity()
        self.visual = ObjectEmbedding(configs.model.visual_feature_dim)
        
        if configs.model.parser.use_splitter:
            from model.nn.parser import ProgramParser
            self.parser = ProgramParser(
                        vocab, gru_hidden_dim= configs.model.parser.hidden_dim,
                        gru_nlayers = configs.model.parser.gru_nlayers,
                        word_emb_dim = configs.model.parser.word_embedding_dim, 
                        positional_emb_dim = configs.model.parser.positional_embedding_dim,
                        use_bert=configs.model.parser.use_bert,
                        discount= configs.discount
            )

            from model.nn.sentence_splitter  import ProgramBuilder
            self.multi_step_builder = ProgramBuilder(program_parser = self.parser ,vocab = self.parser.vocab,  splitter_hidden_dim = configs.model.splitter_hidden_dim,
                                                     splitter_gru_nlayers =  configs.model.splitter_gru_nlayers, splitter_word_emb_dim = configs.model.splitter_word_embedding_dim,
                                                    max_step = configs.model.program_max_step,  use_bert = False, splitter_pos_emb_dim = configs.model.splitter_positional_embedding_dim )
        else:
            from model.nn.multi_step_parser import ProgramParser
            self.parser = ProgramParser(
                        vocab, gru_hidden_dim= configs.model.parser.hidden_dim,
                        gru_nlayers = configs.model.parser.gru_nlayers,
                        word_emb_dim = configs.model.parser.word_embedding_dim, 
                        positional_emb_dim = configs.model.parser.positional_embedding_dim,
                        use_bert=configs.model.parser.use_bert,
                        discount= configs.discount
            )
        from model.nn.program_executor import ProgramExecutor
        self.executor = ProgramExecutor(visual_feature_dim = configs.model.visual_feature_dim[1], 
                         emb_dim = configs.model.emb_dim, nr_actions=gdef.nr_actions,
                         used_concepts_dict = _make_used_concepts_dict(configs.model.make_concept_dict)
                    )

    def prepare_model(self, str_desc):  
        if str_desc == 'usual':
            import model.losses as loss_functions
            self.loss_fn = loss_functions.WeightedSquareIOULoss(box_mode = 'xywh',weights = [100,100,100,100,100,100], reduction = None)
            self.average_iou = loss_functions.IOU2D(box_mode = 'xywh', individual_iou = False, reduction = 'mean')
            self.individual_iou = loss_functions.IOU2D(box_mode ='xywh', individual_iou=True, reduction=None)
            self.soft_precision = loss_functions.IOU2D(box_mode ='xywh', individual_iou=True, reduction=None, soft_precision=True)
            self.hard_precision = loss_functions.HardPrecision(box_mode ='xywh', individual_iou=True, reduction=None)
            self.expectation = loss_functions.ExpectedLoss()
            self.maxloss = loss_functions.MaxLoss()
            self.parser_loss = loss_functions.ParserLoss(reward_signal='loss', reward_offset=self.configs.reward_offset, training_target=self.training_target,)
            self.grounded_program_acc = loss_functions.GroundedProgramAccuracy()
            self.action_acc = loss_functions.ActionProgramAccuracy()
        elif str_desc =='parser_self_supervision':
            import model.losses as loss_functions
            self.expectation = loss_functions.ExpectedLoss()
            self.maxloss = loss_functions.MaxLoss()
            self.parser_loss = loss_functions.ParserLoss(reward_signal='constant', reward_offset = None, training_target=self.training_target,)
        else:
            raise ValueError("Unknown prepare description")
    
    def reset(self, desc):
        if desc =='parser':
            self.parser.reset_parameters()
        elif desc =="action_simulator":
            self.executor.action_sim.reset_parameters()
        else:
            raise ValueError

    def train(self, mode = True, freeze_modules = None):
        super().train(mode)
        from helpers.mytorch.train.freeze import mark_freezed
        if self.training_target == 'parser':
            mark_freezed(self.executor)
            mark_freezed(self.visual)
            mark_freezed(self.resnet)
            mark_freezed(self.multi_step_builder.splitter)
        
        elif self.training_target == 'splitter':
            mark_freezed(self.executor)
            mark_freezed(self.visual)
            mark_freezed(self.resnet)
            mark_freezed(self.parser)
            
        elif self.training_target == 'action_simulator':
            mark_freezed(self.executor.concept_embeddings)
            mark_freezed(self.visual)
            mark_freezed(self.resnet)
            mark_freezed(self.parser)
            mark_freezed(self.multi_step_builder)

        elif self.training_target == 'non_visual':
            mark_freezed(self.executor.concept_embeddings)
            mark_freezed(self.visual)
            mark_freezed(self.resnet)
            mark_freezed(self.multi_step_builder)

        if freeze_modules is not None:
            if 'parser' in freeze_modules:
                mark_freezed(self.parser)
            if 'concept_embeddings' in freeze_modules:
                mark_freezed(self.executor.concept_embeddings)
            if 'action_simulator' in freeze_modules:
                mark_freezed(self.executor.action_sim)
            if 'visual' in freeze_modules:
                mark_freezed(self.visual)
                mark_freezed(self.resnet)
            if 'resnet' in freeze_modules:
                mark_freezed(self.resnet)

    def forward(self, batch_dict,  **kwargs):
        batch_dict = DOView(batch_dict)

        if not kwargs.get('parser_self_supervision', False):
            batch_size = batch_dict.initial_image.size(0) 
            #get visual features
            bboxes_corners = torch.cat([box[:,0:4] for box in batch_dict.initial_bboxes], dim=0)
            image_f = self.resnet(batch_dict.initial_image)
            f_objects = self.visual(image_f, bboxes_corners, batch_dict.object_length) 
        
        #parse the instruction
        if self.training_target == 'concept_embeddings':
            symbolic_programs = batch_dict.program_qsseq
            programs = [dict(scene_id = i, program = p, log_likelihood = torch.tensor(0.0)) for i,p in enumerate(symbolic_programs)]
        elif self.training_target in ['all','non_visual', 'parser']:
            instructions = batch_dict.instruction_lexed if self.parser.use_bert else batch_dict.instruction
            if 'program_parser_candidates_qstree' in batch_dict  and self.parser.training:
                programs = self.parser(instructions, batch_dict.instruction_length, 10, 
                                    batch_dict.attribute_concepts, batch_dict.relational_concepts, batch_dict.action_concepts,
                                    sample_method='sample-enumerate', sample_space=[[v for vs in kv.values() for v in vs] for kv in batch_dict.program_parser_candidates_qstree]
                                    )
            elif 'groundtruth_qstree' in batch_dict and self.parser.training:
                programs = self.parser(instructions, batch_dict.instruction_length, 10, 
                                    batch_dict.attribute_concepts, batch_dict.relational_concepts, batch_dict.action_concepts,
                                    sample_method='groundtruth', sample_space = batch_dict.groundtruth_qstree
                                    )
            else:
                programs = self.parser(instructions, batch_dict.instruction_length, 10, 
                                    batch_dict.attribute_concepts, batch_dict.relational_concepts, batch_dict.action_concepts,
                                    sample_method='sample', sample_space=None, exploration_rate = kwargs.get('exploration_rate',0)
                                    )
        elif self.training_target == 'splitter':
            instructions =  batch_dict.instruction
            programs,num_pairs,is_minimal = self.multi_step_builder(instructions,batch_dict.instruction_length,batch_dict.attribute_concepts, batch_dict.relational_concepts, batch_dict.action_concepts, parser_max_depth = 10)
        else: 
            raise ValueError("Unknown training target")
 
        if not kwargs.get('parser_self_supervision', False):
            #Execute the program
            height,width = self.configs.data.image_shape 
            bboxes_i = normalize_bbox(box_convert(batch_dict.initial_bboxes,'xyxy','xywh'), width, height)
            bboxes_f = normalize_bbox(box_convert(batch_dict.final_bboxes,'xyxy','xywh'), width, height)
            _, g_programs, results, buffers = self.executor(programs, f_objects, bboxes_i,  **kwargs)
            # print("exec:", time.time()-start_time)
            #Find the loss and accuracy
            pred_bboxes = []
            final_bboxes = []
            sids = []
            likelihoods = []
            all_movements = []
            for res in results:
                sid, pred_box, lp, movements = res.values()
                pred_bboxes.append(pred_box)
                all_movements.append(movements)
                final_bboxes.append(bboxes_f[sid])
                sids.append(sid)
                likelihoods.append(lp.clone().detach())

            losses, loss_details = self.loss_fn(pred_bboxes, final_bboxes) 
            iou_individual = self.individual_iou(pred_bboxes, final_bboxes)
            iou_individual = self.expectation(batch_size, sids, likelihoods, iou_individual)
            soft_precision = self.expectation(batch_size, sids, likelihoods, self.soft_precision(pred_bboxes, final_bboxes))
            hard_precision = self.expectation(batch_size, sids, likelihoods, self.hard_precision(pred_bboxes, final_bboxes))
            iou_base_obj = 0
            iou_move_obj = 0
            soft_precision_move_obj = 0 
            hard_precision_move_obj = 0
            iou_move_obj_sqrs = 0
            for id in range(batch_size):
                b_ids  = set([command[2] for command in batch_dict['grounded_program'][id]])
                m_ids  = set([command[1] for command in batch_dict['grounded_program'][id]])
                soft_precision_move_obj += torch.stack([soft_precision[id][m] for m in m_ids]).mean()
                hard_precision_move_obj += torch.stack([hard_precision[id][m] for m in m_ids]).mean()
                iou_base_obj += torch.stack([iou_individual[id][b] for b in b_ids]).mean()
                _tmp = torch.stack([iou_individual[id][m] for m in m_ids]).mean()
                iou_move_obj += _tmp
                iou_move_obj_sqrs += _tmp**2
            soft_precision_move_obj /= batch_size
            hard_precision_move_obj /= batch_size
            iou_base_obj /= batch_size
            iou_move_obj /= batch_size
            iou_move_obj_sqrs /= batch_size

            mean_ious = self.expectation(batch_size, sids, likelihoods, loss_details['ious'])
        #calculate losses 

        if self.training_target == 'concept_embeddings':
            loss = torch.stack(losses).mean()

        elif self.training_target in ['all', 'non_visual']:      
            action_loss = torch.stack(self.expectation(batch_size, sids, likelihoods, losses)).mean()
            # action_loss = torch.stack(losses).mean()
            parser_loss = self.parser_loss(batch_size, programs, losses, **kwargs)
            loss = action_loss + parser_loss.mean() 

        elif self.training_target == 'parser':
            if not kwargs.get('parser_self_supervision', False):
                parser_loss = self.parser_loss(batch_size, programs, losses, **kwargs)
                loss = parser_loss.mean()
            else:
                parser_loss = self.parser_loss(len(programs), programs, None, **kwargs)
                loss = parser_loss.mean()

        elif self.training_target == 'splitter':
            expected_loss_per_example = []
            idx = 0
            for i,num in enumerate(num_pairs):
                program_likelihoods = []
                for j in range(idx,idx+num):
                    program_likelihoods.append(programs[j]['log_likelihood'])
                this_loss  = losses[idx:idx+num]
                
                #to enforce no "idle" in sequence of programs 
                for j in range(num):
                    if not is_minimal[i][j]:
                        this_loss[j] = torch.tensor(100.0, device=this_loss[0].device)

                this_loss = torch.stack(this_loss).detach()*torch.stack(program_likelihoods)
                expected_loss_per_example.append(this_loss.sum())
                idx += num 
            loss =  torch.stack(expected_loss_per_example).mean()

        else:
            raise ValueError('Unknown training target')
     
        if not kwargs.get('parser_self_supervision', False):
            g_prog_acc = self.grounded_program_acc(g_programs, batch_dict.grounded_program)
            # output_dict = {'loss':loss, 'mean_ious': torch.cat(mean_ious), 'indivudual_obj_ious': ( iou_move_obj,iou_base_obj), 'grounded_program_accuracy':np.mean(np.array(g_prog_acc), axis=0), 'action_accuracy': np.mean(np.array(self.action_acc(g_programs, batch_dict.grounded_program)),axis=0)}
            output_dict = {'loss':loss, 'mean_ious': torch.cat(mean_ious), 'move_obj_hp' : hard_precision_move_obj, 'move_obj_sp': soft_precision_move_obj, 'indivudual_obj_ious': ( iou_move_obj,iou_base_obj), 'move_obj_iou_sqrs': iou_move_obj_sqrs, 'grounded_program_accuracy':np.mean(np.array(g_prog_acc), axis=0)}
            output_dict['sym_programs'] = programs 
            output_dict['g_programs'] = g_programs
            output_dict['movements'] = all_movements
            if not self.training:
                output_dict['action_accuracy'] = np.mean(np.array(self.action_acc(g_programs, batch_dict.grounded_program)),axis=0)
        else:
            output_dict = {'loss': loss}
        
        return output_dict

def _make_used_concepts_dict(is_required = False):
    if not is_required:
        return None
    else:
        raise NotImplementedError

