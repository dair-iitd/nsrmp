import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as RNN
from baseline.objectEmbedding import ObjectEmbedding
from baseline.instructionEmbedding import InstructionEmbedding
from baseline.baselineModel import BaselineModelOld
from helpers.utils.container import DOView
from nsrmp.datasets.vocab import Vocab
import nsrmp.model.losses as loss_functions
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.mytorch.vision.ops.boxes import box_convert, normalize_bbox

class BaselineModelExecutor(nn.Module):
    def __init__(self, vocab, configs):
        super().__init__()
        self.vocab = vocab
        self.configs = configs
        self.embed = nn.Embedding(len(vocab), 256, padding_idx=0)
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.single_step = BaselineModelOld(vocab, configs)
        self.loss_fn = loss_functions.WeightedSquareIOULoss(box_mode = 'xywh', weights = [100,100,100,100,90,50], reduction = None)
        self.iou_2D = loss_functions.IOU2D(box_mode = 'xywh', individual_iou = False, reduction = 'sum')
    
    def forward(self, batch_dict, batch=0):
        batch_dict = DOView(batch_dict)
        batch_size = batch_dict.initial_image.size(0)
        bboxes_corners = [box[:, 0:4] for box in batch_dict.initial_bboxes]
        f_objects = self.single_step.visual(batch_dict.initial_image, bboxes_corners, batch_dict.object_length)
        
        max_length = batch_dict.instruction.size(1)

        height, width = self.configs.data.image_shape 
        bboxes_i = normalize_bbox(box_convert(batch_dict.initial_bboxes,'xyxy','xywh'), width, height)
        bboxes_f = normalize_bbox(box_convert(batch_dict.final_bboxes,'xyxy','xywh'), width, height)

        pred_bboxes, final_bboxes = [], []

        # find breakpoint in the natural-language sentence
        init = False
        object_count = f_objects[batch].size(0)
        instructions = []
        self.decompose(batch_dict.instruction[batch][0:batch_dict.instruction_length[batch]], instructions, max_length)
        
        num_steps = len(instructions)
        bbox_list = []
        for t in range(num_steps):
            _, f_instruction = self.single_step.language(instructions[t].unsqueeze(0), torch.tensor([instructions[t].size(0)], device=torch.device('cuda')))
            f_action = F.softmax(self.single_step.decoder(f_instruction), dim=-1)

            subject_instruction = (self.single_step.subject_map(f_instruction)).repeat(object_count, 1)
            object_instruction = (self.single_step.object_map(f_instruction)).repeat(object_count, 1)
            subject_attention = F.softmax(self.single_step.attention_sub(torch.cat([subject_instruction, f_objects[batch]], dim=-1)), dim=0)
            object_attention = F.softmax(self.single_step.attention_obj(torch.cat([object_instruction, f_objects[batch]], dim=-1)), dim=0)
            
            subject_shape_instruction = (self.single_step.subject_shape_map(f_instruction)).repeat(object_count, 1)
            object_shape_instruction = (self.single_step.object_shape_map(f_instruction)).repeat(object_count, 1)
            subject_shape_attention = F.softmax(self.single_step.attention_shape_sub(torch.cat([subject_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
            object_shape_attention = F.softmax(self.single_step.attention_shape_obj(torch.cat([object_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
            
            total_subject_attention = subject_attention * subject_shape_attention
            total_subject_attention = (total_subject_attention) / torch.sum(total_subject_attention, dim=0)
            total_object_attention = object_attention * object_shape_attention
            total_object_attention = (total_object_attention) / torch.sum(total_object_attention, dim=0)

            o1, o2 = torch.argmax(total_subject_attention).item(), torch.argmax(total_object_attention).item()

            subject_loc = bboxes_i[batch][o1]
            object_loc = bboxes_i[batch][o2]
            f_embedding = torch.cat([f_action[0], object_loc, subject_loc]).unsqueeze(0)
            pred = self.single_step.pred(f_embedding)
            original_subject_loc = subject_loc.clone()
            bboxes_i[batch][o1] = pred
            bbox_list.append((pred.clone(), o1, original_subject_loc))
        return bbox_list
        
    def decompose(self, instruction, instructions, max_length):
        if instruction.size(0) == 2:
            return
        instruction = instruction.clone()
        length = torch.tensor([instruction.size(0)], device=torch.device('cpu'))
        instruction = F.pad(instruction, (0, max_length - instruction.size(0)))
        e_instruction = self.embed(instruction.unsqueeze(0))
        p_instructions = RNN.pack_padded_sequence(e_instruction[:, 1:], length - 2, batch_first=True, enforce_sorted=False)
        lstm_output = self.lstm(p_instructions)
        linear_output = RNN.PackedSequence(data=self.linear(lstm_output[0].data), batch_sizes=lstm_output[0].batch_sizes, sorted_indices=lstm_output[0].sorted_indices, unsorted_indices=lstm_output[0].unsorted_indices)
        linear_output = RNN.pad_packed_sequence(linear_output, batch_first=True, padding_value=-float('inf'))
        probs = F.softmax(linear_output[0][:, :min(linear_output[0].size(1), 30)], dim=-2)
        bp = torch.argmax(probs).item()
        if bp < length - 3:
            # two sentences
            t_instruction = instruction[:bp + 2].clone()
            t_instruction[bp + 1] = 2
            self.decompose(t_instruction, instructions, max_length)
            s_instruction = instruction[bp + 1:length].clone()
            s_instruction[0] = 3
            self.decompose(s_instruction, instructions, max_length)
        else:
            # one sentence
            t_instruction = instruction.clone()
            instructions.append(t_instruction)