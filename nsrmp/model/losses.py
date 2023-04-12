#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   losses.py
#Time    :   2022/06/28 02:00:02
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

import random
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from helpers.mytorch.vision.ops.boxes import box_convert
from datasets.common.program_translator import nsrmseq_to_nsrmtree
from datasets.common.program_analysis import nsrmtree_to_string

def identity(x):
    return x

def _get_reduction_fn(identifier):
    if identifier is None:
        return identity
    reduction_fn = getattr(torch, identifier, None)
    assert reduction_fn is not None, 'Given reduction is not applicable'
    return reduction_fn

class HardPrecision(nn.Module):
    def __init__(self, box_mode = 'xywh', individual_iou = True, reduction:str = None):
        super().__init__()
        self.box_mode = box_mode
        self.individual_iou = individual_iou
        self.reduction = reduction
        self.reduction_fn = _get_reduction_fn(self.reduction)
        
    def forward(self,pred_bboxes, final_bboxes):
        '''
        Note: The bboxes should be in [x1,y1,x2,y2] format. (x1,y1) is the left top co-ordinate.

        Inputs:
            pred_bboxes: (Tensor[B,N,4] or List[Tensor[M,4]]). Here, N is the number of objects/classes.
                            The predicted bboxes
            final_bboxes: (Tensor[B,N,4] or List[Tensor[M,4]]). The ground-truth bboxes
        Return:
            ious:(Tensor[B,N] or List[Tensor[M]]) 0-1 ious between corresponding boxxes. If center of the predicted bbox lies inside gold box then 1 else 0
                 If individual_iou = False, then the ious are averaged per image. In that case, the output is List[] or Tensor[B]
                  If reduction = 'mean' or 'sum', the respective operation is applied over all values and a single float tensor will be returned
         '''
        if torch.is_tensor(pred_bboxes):
            ious = self._iou_tensor(pred_bboxes,final_bboxes)
            return self.reduction_fn(ious)
            
        elif type(pred_bboxes) == list:
            ious = [self._iou_tensor(pred_bboxes[i],final_bboxes[i])  for i in range(len(pred_bboxes)) ]
            if self.reduction is not None: 
                ious = self.reduction_fn(torch.stack(ious))
            return ious
        else:
            raise NotImplementedError

    def _iou_tensor(self, pred_bboxes, final_bboxes):
        ious = self._hard_inter(pred_bboxes,final_bboxes)
        if not self.individual_iou:
            ious = torch.mean(ious, dim = -1)
        return ious
 

    def _hard_inter(self, _bbox1, _bbox2):
        if self.box_mode == 'xywh':
            bbox1 = box_convert(_bbox1,'xywh','xyxy')
            bbox2 = box_convert(_bbox2, 'xywh', 'xyxy')
        elif self.box_mode == 'xyxy':
            bbox1, bbox2 = _bbox1, _bbox2
        else:
            raise NotImplementedError

        #at this point the bbox1 and bbox2 will be in 'xyxy' mode
        s = [i for i in _bbox1.shape]
        s[-1] = 2
        center = torch.zeros(s, device = _bbox1.device)
        center[...,0] = (bbox1[...,0]+bbox1[...,2])/2
        center[...,1] = (bbox1[...,1]+bbox2[...,3])/2
        is_x_inside = torch.logical_and(bbox2[...,0] <= center[...,0], bbox2[...,2] >= center[...,0])
        is_y_inside = torch.logical_and(bbox2[...,1] <= center[...,1], bbox2[...,3] >= center[...,1])
        hard_inter = torch.logical_and(is_x_inside, is_y_inside)
        return hard_inter
    

class IOU2D(nn.Module):
    def __init__(self, box_mode = 'xywh', individual_iou = True, reduction:str = None, soft_precision = False):
        super().__init__()
        self.box_mode = box_mode
        self.individual_iou = individual_iou
        self.reduction = reduction
        self.reduction_fn = _get_reduction_fn(self.reduction)
        self.soft_precision = soft_precision
    def forward(self,pred_bboxes, final_bboxes):
        '''
        Note: The bboxes should be in [x1,y1,x2,y2] format. (x1,y1) is the left top co-ordinate.

        Inputs:
            pred_bboxes: (Tensor[B,N,4] or List[Tensor[M,4]]). Here, N is the number of objects/classes.
                            The predicted bboxes
            final_bboxes: (Tensor[B,N,4] or List[Tensor[M,4]]). The ground-truth bboxes
            soft_precision: If true, inter/ true_bbox_area is calculated. Else, inter/union is calculated
        Return:
            ious:(Tensor[B,N] or List[Tensor[M]]) ious between corresponding boxxes. 
                 If individual_iou = False, then the ious are averaged per image. In that case, the output is List[] or Tensor[B]
                  If reduction = 'mean' or 'sum', the respective operation is applied over all values and a single float tensor will be returned
         '''
        if torch.is_tensor(pred_bboxes):
            ious = self._iou_tensor(pred_bboxes,final_bboxes)
            return self.reduction_fn(ious)
            
        elif type(pred_bboxes) == list:
            ious = [self._iou_tensor(pred_bboxes[i],final_bboxes[i])  for i in range(len(pred_bboxes)) ]
            if self.reduction is not None: 
                ious = self.reduction_fn(torch.stack(ious))
            return ious
        else:
            raise NotImplementedError

    def _iou_tensor(self, pred_bboxes, final_bboxes):
        union, inter = self._union_and_inter(pred_bboxes,final_bboxes)
        if self.soft_precision:
            union = self._area(final_bboxes, box_mode=self.box_mode)
        ious = torch.div(inter,union)
        if not self.individual_iou:
            ious = torch.mean(ious, dim = -1)
        return ious
 
    def _area(self, bbox, box_mode):
        '''
        bbox: Tensor[...,4]. 
        
        Sometimes, the intersection may be empty. In that case 0<=x1<=x2 or 0<=y1<=y2 will not be satisfied.
        So, if the constraint is not satisfied, we return 0 as the area.
        '''
        if box_mode == 'xyxy':
            correction_sign = F.relu(torch.sign(bbox[...,2]-bbox[...,0]))*F.relu(torch.sign(bbox[...,3]-bbox[...,1]))
            return (bbox[...,0]-bbox[...,2])* (bbox[...,1]-bbox[...,3]) * correction_sign
        elif box_mode == 'xywh':
            correction_sign = F.relu(torch.sign(bbox[...,2])) * F.relu(torch.sign(bbox[...,3]))
            return bbox[...,2]*bbox[...,3]*correction_sign
        else:
            raise NotImplementedError

    def _union_and_inter(self, _bbox1, _bbox2):
        if self.box_mode == 'xywh':
            bbox1 = box_convert(_bbox1,'xywh','xyxy')
            bbox2 = box_convert(_bbox2, 'xywh', 'xyxy')
        elif self.box_mode == 'xyxy':
            bbox1, bbox2 = _bbox1, _bbox2
        else:
            raise NotImplementedError

        #at this point the bbox1 and bbox2 will be in 'xyxy' mode
        inter = torch.zeros_like(bbox1)
        inter[...,0] = torch.max(bbox1[...,0],bbox2[...,0])
        inter[...,1] = torch.max(bbox1[...,1],bbox2[...,1])
        inter[...,2] = torch.min(bbox1[...,2], bbox2[...,2])
        inter[...,3] = torch.min(bbox1[...,3], bbox2[...,3])
        inter_area = self._area(inter, 'xyxy')
        return self._area(bbox1,'xyxy') + self._area(bbox2,'xyxy') - inter_area, inter_area



class IOU3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError


class MaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch_size, sids, log_likelihoods, losses):
        '''
        Inputs:
                sid:(List[Tensor]) The scene ids.
                likelihoods: (List[Tensor]) The corresponding likelihoods of the loss/program
                losses: (List[Tensor])
        Output:
                exp_losses: List[Tensor]. The expected loss where the expectation is taken over the losses of the same scene_id. The length of the list is the batch_size
        '''
        max_losses = []
        for i in range(batch_size):
            log_likelihood = [log_likelihoods[j] for j in range(len(sids)) if sids[j] == i ]
            if len(log_likelihood) == 0:
                continue
            log_likelihood = torch.stack(log_likelihood)
            this_scene_loss = [losses[j] for j in range(len(sids)) if sids[j] == i ]    
            max_losses.append(this_scene_loss[log_likelihood.argmax()])
        return max_losses


class ExpectedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch_size, sids, log_likelihoods, losses):
        '''
        Inputs:
                sid:(List[Tensor]) The scene ids.
                likelihoods: (List[Tensor]) The corresponding likelihoods of the loss/program
                losses: (List[Tensor])
        Output:
                exp_losses: List[Tensor]. The expected loss where the expectation is taken over the losses of the same scene_id. The length of the list is the batch_size
        '''
        exp_losses = []
        for i in range(batch_size):
            log_likelihood = [log_likelihoods[j] for j in range(len(sids)) if sids[j] == i ]
            if len(log_likelihood) == 0:
                continue
            likelihood = F.softmax(torch.stack(log_likelihood), dim = -1)
            this_scene_loss = [losses[j] for j in range(len(sids)) if sids[j] == i ]
            loss = torch.stack([likelihood[j]*this_scene_loss[j] for j in range(likelihood.size(0))]).sum(dim=0)
            exp_losses.append(loss)
        assert len(exp_losses) == batch_size
        return exp_losses


class WeightedSquareIOULoss(nn.Module):
    def __init__(self, box_mode = 'xywh', iou_type = '2d', weights = None, reduction = None):
        super().__init__()
        self.box_mode = box_mode
        self.weights = [10,10,10,10,4,2] if weights is None else weights
        assert iou_type in ['2d','3d'], 'Unknown iou type. Expected 2d or 3d but given {}'.format(iou_type)
        self.reduction = reduction
        if self.reduction is not None:
            self.reduction_fn = _get_reduction_fn(self.reduction)
        self.iou = IOU2D(box_mode = self.box_mode, individual_iou= False) if iou_type == '2d' else IOU3D()

    def forward(self, pred_bboxes, final_bboxes):
        '''
        Inputs:
            pred_bboxes: (Tensor[B,N,4] or List[Tensor[M,4]]). Here, N is the number of objects/classes.
                            The predicted bboxes
            final_bboxes: (Tensor[B,N,4] or List[Tensor[M,4]]). The ground-truth bboxes
        
        Return:
            losses:(Tensor[B] or List[]) weighted mean-square and iou losses between corresponding boxxes. 
                   If reduction = 'mean' or 'sum', the respective operation is applied over all values and a single float tensor will be returned
            details: Dict(). The keys of the dict are mse and ious.  
         '''
        if torch.is_tensor(pred_bboxes):
            losses, details = self._loss_tensor(pred_bboxes,final_bboxes)
            if self.reduction is not None:
                return self.reduction_fn(losses), details
            else:
                return losses, details

        elif type(pred_bboxes) == list:
            from collections import defaultdict
            losses, details = [], defaultdict(list)
            for i in range(len(pred_bboxes)):
                loss, detail = self._loss_tensor(pred_bboxes[i].unsqueeze(0),final_bboxes[i].unsqueeze(0))
                losses.append(loss)
                for k in detail.keys():
                    details[k].append(detail[k]) 
            if self.reduction is not None:
                losses = self.reduction_fn(torch.tensor(losses))
            return losses, details
        else:
            raise NotImplementedError

    def _loss_tensor(self, _bbox1, _bbox2):
        if self.box_mode == 'xyxy':
            bbox1, bbox2 = box_convert(_bbox1, 'xyxy', 'xywh'), box_convert(_bbox2, 'xyxy', 'xywh')
        elif self.box_mode == 'xywh':
            bbox1, bbox2 = _bbox1, _bbox2
        else: 
            raise NotImplementedError
        ious = self.iou(_bbox1, _bbox2)
        mse = torch.square((bbox1 - bbox2).abs()).sum(dim=-2) 
        return (torch.tensor(self.weights, device = _bbox1.device)*torch.cat([mse.squeeze(),1-ious], dim=-1)).sum(), dict(mse = mse, ious = ious)


class GroundedProgramAccuracy(object):
    def __init__(self):
        super().__init__()

    def __call__(self, pred_programs, original_programs):
        results = []
        for j in range(len(original_programs)):
            accuracy1 = 0
            accuracy2 = 0
            pred_progs = [p['grounded_program'] for p in pred_programs if p['scene_id'] == j]
            log_likelihoods = [p['log_likelihood'] for p in pred_programs if p['scene_id'] == j]
            if len(log_likelihoods) == 0:
                continue
            log_likelihoods = torch.stack(log_likelihoods, dim = 0)
            likelihoods = F.softmax(log_likelihoods, dim=-1)
            goal= original_programs[j]
            for pred,p in zip(pred_progs,likelihoods):
                acc1=0
                acc2=0 
                if len(pred) == len(goal):
                    for i in range(len(pred)):
                            acc1 += int(pred[i][1].argmax().item() == goal[i][1])
                            acc2 += + int(pred[i][2].argmax().item() == goal[i][2])
                    acc1 = 0 if acc1 != len(goal) else 1
                    acc2 = 0 if acc2 != len(goal) else 1
                accuracy1 += acc1*p
                accuracy2 += acc2*p 
            results.append([accuracy1.item(),accuracy2.item()])
        return results

class ActionProgramAccuracy(object):
    def __init__(self):
        super().__init__()

    def __call__(self, pred_programs, original_programs):
        results = []
        for j in range(len(original_programs)):
            pred_progs = [p['grounded_program'] for p in pred_programs if p['scene_id'] == j]
            log_likelihoods = [p['log_likelihood'] for p in pred_programs if p['scene_id'] == j]
            if len(log_likelihoods) == 0:
                continue
            log_likelihoods = torch.stack(log_likelihoods, dim = 0)
            likelihoods = F.softmax(log_likelihoods, dim=-1)
            goal= original_programs[j]
            acc0 = 0 
            assert len(pred_progs) == 1
            for pred,p in zip(pred_progs,likelihoods):
                acc0 =0 
                if len(pred) == len(goal):
                    # breakpoint()
                    for i in range(len(pred)):
                            if pred[i][0].lower() == goal[i][0].lower():
                                acc0 += 1 
                    acc0 = acc0/len(pred)
                    results.append(acc0)
                    assert acc0 <= 1 
                    acc0 = 0 
                else: 
                    results.append(0)
            
        return results


class GroundedProgramAccuracy(object):
    def __init__(self):
        super().__init__()

    def __call__(self, pred_programs, original_programs):
        results = []
        for j in range(len(original_programs)):
            accuracy1 = 0
            accuracy2 = 0
            pred_progs = [p['grounded_program'] for p in pred_programs if p['scene_id'] == j]
            log_likelihoods = [p['log_likelihood'] for p in pred_programs if p['scene_id'] == j]
            if len(log_likelihoods) == 0:
                continue
            log_likelihoods = torch.stack(log_likelihoods, dim = 0)
            likelihoods = F.softmax(log_likelihoods, dim=-1)
            goal= original_programs[j]
            for pred,p in zip(pred_progs,likelihoods):
                acc1=0
                acc2=0 
                if len(pred) == len(goal):
                    for i in range(len(pred)):
                            acc1 += int(pred[i][1].argmax().item() == goal[i][1])
                            acc2 += + int(pred[i][2].argmax().item() == goal[i][2])
                    acc1 = 0 if acc1 != len(goal) else 1
                    acc2 = 0 if acc2 != len(goal) else 1
                accuracy1 += acc1*p
                accuracy2 += acc2*p 
            results.append([accuracy1.item(),accuracy2.item()])
        return results


        
class SymbolicProgramAccuracy(object):
    def  __init__(self):
        super().__init__()

    def __call__(self, pred_programs, original_programs, group_filter = False):
        results = []
        for j in range(len(original_programs)):
            accuracy = 0
            pred_progs = [p['program'] for p in pred_programs if p['scene_id'] == j]
            log_likelihoods = [p['log_likelihood'] for p in pred_programs if p['scene_id'] == j]
            if len(log_likelihoods) == 0:
                continue
            log_likelihoods = torch.stack(log_likelihoods, dim = 0)
            likelihoods = F.softmax(log_likelihoods, dim=-1)
            goal_prog = nsrmtree_to_string(nsrmseq_to_nsrmtree(original_programs[j]))
            goal_prog = re.findall('\w+', goal_prog)
            if group_filter:
                goal_prog = [w for i, w in enumerate(goal_prog) if w!='filter' or goal_prog[i-1] != 'filter']
            for pred,p in zip(pred_progs, likelihoods):
                pred = nsrmseq_to_nsrmtree(pred)
                pred = nsrmtree_to_string(pred)
                # if j==0:
                #     print(pred)
                pred = re.findall('\w+', pred)
                accuracy += int(pred == goal_prog)*p
            results.append(accuracy)
        return results
            

                
            

class ParserLoss(nn.Module):
    def __init__(self, reward_signal , reward_offset, training_target):
        super().__init__()
        self.reward_signal = reward_signal
        self.reward_offset = reward_offset
        self.training_target = training_target

    def _resolve_conflict(self, progs, resolve_method = 'ensure_sequentiality'):
        if resolve_method == "random":
            return random.choice([i for i in range(len(progs))])
        elif resolve_method == "ensure_sequentiality":
            from datasets.common.program_analysis import concepts_in_nsrmtree
            concepts_dicts = [concepts_in_nsrmtree(nsrmseq_to_nsrmtree(p)) for p in progs]
            choosen_idx = 0
            for cur_idx in range(1,len(progs)):
                flag = False
                for k in concepts_dicts[choosen_idx].keys():
                    if concepts_dicts[cur_idx][k] < concepts_dicts[choosen_idx][k]:
                        flag = True
                        break
                if flag:
                    choosen_idx = cur_idx
            return choosen_idx
        else:
            raise ValueError("Unknown resolve method to break conflicts in prog")

    def forward(self,batch_size, programs_pd, loss, baseline = True, temperature = 1, **kwargs):
        policy_loss = 0
        for i in range(batch_size):
            log_likelihoods = [p['log_likelihood'] for p in programs_pd if i == p['scene_id']]
            
            if len(log_likelihoods) == 0:
                continue
        
            log_likelihoods = torch.stack(log_likelihoods, dim = 0)
            discounted_log_likelihood = torch.stack([p['discounted_log_likelihood'] for p in programs_pd if i == p['scene_id']], dim =0)
            if self.reward_signal == 'loss':
                if baseline:
                    if self.training_target in ['parser', 'splitter']:
                        this_scene_losses = torch.stack([loss[j] for j,p in enumerate(programs_pd) if i == p['scene_id']], dim = 0)
                        min_idx = this_scene_losses.argmin()
                        this_scene_losses = this_scene_losses -  this_scene_losses.min()
                        conflict = torch.sum(this_scene_losses <= 0.10) > 1
                        if conflict:
                            this_scene_prog = [p for p in programs_pd if i == p['scene_id']]
                            conflict_prog_ids = [i for i,l in enumerate(this_scene_losses) if l <=0.10]
                            choosen_prog_idx = self._resolve_conflict([p['program'] for idx,p in enumerate(this_scene_prog) if idx in conflict_prog_ids])
                            rewards = torch.zeros(len(this_scene_losses), device=discounted_log_likelihood.device) 
                            rewards[conflict_prog_ids[choosen_prog_idx]] = 10
                        else:
                            rewards = torch.zeros(len(this_scene_losses), device=discounted_log_likelihood.device)
                            rewards[min_idx] = 10
                    elif self.training_target in ['all', 'visual']:
                        this_scene_losses = torch.stack([loss[j] for j,p in enumerate(programs_pd) if i == p['scene_id']], dim = 0)
                        rewards = (this_scene_losses.max()+this_scene_losses.min())/2 - this_scene_losses

                    else:
                        raise ValueError("training target not supported")
                else:
                    rewards = 100*F.relu(self.reward_offset - torch.stack([loss[j] for j,p in enumerate(programs_pd) if i == p['scene_id']], dim = 0))
                
            elif self.reward_signal == 'constant':
                rewards = 10*torch.ones(discounted_log_likelihood.size(), device=discounted_log_likelihood.device)
            
            likelihood = F.softmax(log_likelihoods/temperature, dim=-1)
            policy_loss += (-(likelihood * rewards).detach() * discounted_log_likelihood).sum()
        return policy_loss/batch_size
