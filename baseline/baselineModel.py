######################
### Baseline Model ###
######################

# PyTorch related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as RNN

# Helper imports
from helpers.utils.container import DOView
import nsrmp.model.losses as loss_functions
from helpers.mytorch.cuda.copy import async_copy_to
from helpers.mytorch.vision.ops.boxes import box_convert, normalize_bbox

# Baseline related imports
from baseline.objectEmbedding import ObjectEmbedding
from baseline.instructionEmbedding import InstructionEmbedding


# Helper module used in BaselineModel
class Map(nn.Module):
    
    # Arguments:
    #   input_dim: (int): Dimension of input
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
    
    # Arguments:
    #   batch: (tensor(N, 1024)): Input dense embedding
    # Return Value:
    #   output: (tensor(N, 256)): Output dense embedding
    # Here, N = number of dense embeddings
    def forward(self, batch):
        output = self.model(batch)
        return output


# Baseline module without splitter
class BaselineModel(nn.Module):
    
    # Arguments:
    #   vocab: (Vocab)   : Word vocabulary (vocab.json)
    #   configs: (DOView): Baseline configurations (configs.py)
    def __init__(self, vocab, configs):
        super().__init__()
        self.vocab = vocab
        self.configs = configs
        self.data_assoc = configs.model.data_assoc
        self.visual = ObjectEmbedding(configs.data.image_shape, configs.model.visual_feature_dim, configs.data.object_feature_bbox_size)
        self.language = InstructionEmbedding(vocab, configs.model.parser_hidden_dim, configs.model.gru_nlayers, configs.model.word_embedding_dim, configs.model.positional_embedding_dim)
        self.language_encoder_dim = configs.model.parser_hidden_dim * configs.model.gru_nlayers
        
        ######################
        ### Action Decoder ###
        ######################
        self.decoder = nn.Sequential(
            nn.Linear(self.language_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        #########################
        ### Subject Attention ###
        #########################
        self.subject_map = Map(self.language_encoder_dim)
        self.attention_sub = nn.Sequential(
            nn.Linear(configs.model.visual_feature_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.subject_shape_map = Map(self.language_encoder_dim)
        self.attention_shape_sub = nn.Sequential(
            nn.Linear(configs.model.visual_feature_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        ########################
        ### Object Attention ###
        ########################
        self.object_map = Map(self.language_encoder_dim)
        self.attention_obj = nn.Sequential(
            nn.Linear(configs.model.visual_feature_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.object_shape_map = Map(self.language_encoder_dim)
        self.attention_shape_obj = nn.Sequential(
            nn.Linear(configs.model.visual_feature_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        ########################
        ### Action Simulator ###
        ########################
        self.pred = nn.Sequential(
            nn.Linear(13, 256), 
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh()
        )
        if configs.model.use_iou_loss:
            self.loss_fn = loss_functions.WeightedSquareIOULoss(box_mode="xywh", weights=[100, 100, 100, 100, 90, 50], reduction=None)
        else:
            self.loss_fn = loss_functions.WeightedSquareIOULoss(box_mode="xywh", weights=[100, 100, 100, 100, 90, 0], reduction=None)
        self.iou_2D = loss_functions.IOU2D(box_mode="xywh", individual_iou=False, reduction='sum')

    # Arguments:
    #   batch_dict: (Dictionary): Input data
    # Return Value:
    #   loss: (tensor())          : Loss value
    #   loss_details: (Dictionary): Details of loss (MSE|IoU)
    #   mean_sub_attn: (float)    : Mean correct subject attention value
    #   mean_obj_attn: (float)    : Mean correct object attention value
    def forward(self, batch_dict):
        batch_dict = DOView(batch_dict)
        batch_size = batch_dict.initial_image.size(0)
        height, width = self.configs.data.image_shape 
        bboxes_corners = [box[:, 0:4] for box in batch_dict.initial_bboxes]
        bboxes_corners_final = [box[:, 0:4] for box in batch_dict.final_bboxes]
        f_objects = self.visual(batch_dict.initial_image, bboxes_corners, batch_dict.object_length)
        bboxes_i = normalize_bbox(box_convert(batch_dict.initial_bboxes,'xyxy','xywh'), width, height)
        bboxes_f = normalize_bbox(box_convert(batch_dict.final_bboxes,'xyxy','xywh'), width, height)
        
        if not self.data_assoc:
            with torch.no_grad():
                f_objects_final = self.visual(batch_dict.final_image, bboxes_corners_final, batch_dict.object_length)
            mapping_acc = torch.zeros(batch_size)
            for batch in range(batch_size):
                inp_emb = f_objects[batch]
                outp_emb = f_objects_final[batch]
                num_objects = inp_emb.shape[0]
                inp_emb = inp_emb.unsqueeze(1).repeat(1, num_objects, 1) # N X N X d, repetition along columns
                outp_emb = outp_emb.unsqueeze(0).repeat(num_objects, 1, 1) # N X N X d, repetition along rows

                rho = F.cosine_similarity(inp_emb, outp_emb, dim=2) # Shape NxN
                correct = 0
                bboxes_final = async_copy_to(torch.zeros(bboxes_f[batch].shape), dev=0, main_stream=None)  # num_objects x 5
                selection = async_copy_to(torch.zeros(num_objects), dev=0, main_stream=None)
                for i in range(num_objects):
                    new_selection = F.gumbel_softmax(rho[i] + (-100.0*selection), hard=True, tau=1e9)
                    correct += (new_selection.nonzero().item() == i)
                    bboxes_final[i] += torch.sum(new_selection.view(num_objects,1) * bboxes_f[batch], dim=0)
                    selection += new_selection 
                bboxes_f[batch] = bboxes_final 
                mapping_acc[batch] = correct/num_objects
            mapping_acc = torch.mean(mapping_acc)
        
        f_instruction = self.language(batch_dict.instruction, batch_dict.instruction_length)
        f_action = F.softmax(self.decoder(f_instruction), dim=-1)
        
        init = False
        pred_bboxes = []
        final_bboxes = []
        attention_scores, object_scores = [], []
        sub_attn, obj_attn = [], []
        for batch in range(batch_size):
            object_count = f_objects[batch].size(0)
            subject_instruction = (self.subject_map(f_instruction[batch])).repeat(object_count, 1)
            subject_attention = F.softmax(self.attention_sub(torch.cat([subject_instruction, f_objects[batch]], dim=-1)), dim=0)
            subject_shape_instruction = (self.subject_shape_map(f_instruction[batch])).repeat(object_count, 1)
            subject_shape_attention = F.softmax(self.attention_shape_sub(torch.cat([subject_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
            total_subject_attention = subject_attention * subject_shape_attention
            total_subject_attention = (total_subject_attention) / torch.sum(total_subject_attention, dim=0)

            object_instruction = (self.object_map(f_instruction[batch])).repeat(object_count, 1) 
            object_attention = F.softmax(self.attention_obj(torch.cat([object_instruction, f_objects[batch]], dim=-1)), dim=0)
            object_shape_instruction = (self.object_shape_map(f_instruction[batch])).repeat(object_count, 1)
            object_shape_attention = F.softmax(self.attention_shape_obj(torch.cat([object_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
            total_object_attention = object_attention * object_shape_attention
            total_object_attention = (total_object_attention) / torch.sum(total_object_attention, dim=0)

            sub_attn.append(total_subject_attention[batch_dict.grounded_program[batch][0][1]][0].item())
            obj_attn.append(total_object_attention[batch_dict.grounded_program[batch][0][2]][0].item())
            attention_scores.append(total_subject_attention)
            object_scores.append(total_object_attention)
            
            subject_loc = torch.sum(total_subject_attention * bboxes_i[batch], dim=0)
            object_loc = torch.sum(total_object_attention * bboxes_i[batch], dim=0)
            f_embedding = torch.cat([f_action[batch], object_loc, subject_loc]).unsqueeze(0)
            if init:
                final_embedding = torch.cat([final_embedding, f_embedding], dim=0)
            else:
                final_embedding = f_embedding
                init = True
        pred = self.pred(final_embedding)
        for batch in range(batch_size):
            pred_bbox =  (1 - attention_scores[batch]) * bboxes_i[batch] + attention_scores[batch] * pred[batch]
            pred_bboxes.append(pred_bbox)
            final_bboxes.append(bboxes_f[batch])
        losses, loss_details = self.loss_fn(pred_bboxes, final_bboxes)
        loss = torch.stack(losses).mean()
        mean_sub_attn = torch.tensor(sub_attn).mean().item()
        mean_obj_attn = torch.tensor(obj_attn).mean().item()
        return loss, loss_details, mean_sub_attn, mean_obj_attn

    # Arguments:
    #   batch_dict: (Dictionary): Input data
    # Return Value:
    #   argO1: (int)                  : Count of correct subject identification
    #   argO2: (int)                  : Count of correct object identification
    #   individual_iou: (tensor())    : Sum of moved object IoU
    #   individual_iou_var: (tensor()): Sum of squares of moved object IoU
    #   total_iou: (tensor())         : Sum of all object IoU
    def inference(self, batch_dict):
        batch_dict = DOView(batch_dict)
        batch_size = batch_dict.initial_image.size(0)
        bboxes_corners = [box[:, 0:4] for box in batch_dict.initial_bboxes]
        f_objects = self.visual(batch_dict.initial_image, bboxes_corners, batch_dict.object_length)
        f_instruction = self.language(batch_dict.instruction, batch_dict.instruction_length)
        f_action = F.softmax(self.decoder(f_instruction), dim=-1)
        height, width = self.configs.data.image_shape 
        bboxes_i = normalize_bbox(box_convert(batch_dict.initial_bboxes,'xyxy','xywh'), width, height)
        bboxes_f = normalize_bbox(box_convert(batch_dict.final_bboxes,'xyxy','xywh'), width, height)
        init = False
        attention_scores = []
        m_object = []
        argO1, argO2 = 0, 0
        for batch in range(batch_size):
            object_count = f_objects[batch].size(0)
            subject_instruction = (self.subject_map(f_instruction[batch])).repeat(object_count, 1)
            subject_attention = F.softmax(self.attention_sub(torch.cat([subject_instruction, f_objects[batch]], dim=-1)), dim=0)
            subject_shape_instruction = (self.subject_shape_map(f_instruction[batch])).repeat(object_count, 1)
            subject_shape_attention = F.softmax(self.attention_shape_sub(torch.cat([subject_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
            total_subject_attention = subject_attention * subject_shape_attention
            total_subject_attention = (total_subject_attention) / torch.sum(total_subject_attention, dim=0)

            object_instruction = (self.object_map(f_instruction[batch])).repeat(object_count, 1)
            object_attention = F.softmax(self.attention_obj(torch.cat([object_instruction, f_objects[batch]], dim=-1)), dim=0)
            object_shape_instruction = (self.object_shape_map(f_instruction[batch])).repeat(object_count, 1)            
            object_shape_attention = F.softmax(self.attention_shape_obj(torch.cat([object_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
            total_object_attention = object_attention * object_shape_attention
            total_object_attention = (total_object_attention) / torch.sum(total_object_attention, dim=0)

            m_object.append(torch.argmax(total_subject_attention).item())
            attention_scores.append(total_subject_attention)
            if batch_dict.grounded_program[batch][0][1] == m_object[batch]:
                argO1 += 1
            if batch_dict.grounded_program[batch][0][2] == torch.argmax(total_object_attention).item():
                argO2 += 1
            subject_loc = bboxes_i[batch][m_object[batch]]
            object_loc = bboxes_i[batch][torch.argmax(total_object_attention).item()]
            f_embedding = torch.cat([f_action[batch], object_loc, subject_loc]).unsqueeze(0)
            if init:
                final_embedding = torch.cat([final_embedding, f_embedding], dim=0)
            else:
                final_embedding = f_embedding
                init = True
        pred = self.pred(final_embedding)
        pred_bboxes, final_bboxes, m_pred_bboxes, m_final_bboxes = [], [], [], []
        individual_iou_var = 0.0
        for batch in range(batch_size):
            pred_bbox = bboxes_i[batch].clone()
            pred_bbox[m_object[batch]] = pred[batch]
            pred_bboxes.append(pred_bbox)
            final_bboxes.append(bboxes_f[batch])
            if batch_dict.grounded_program[batch][0][1] != m_object[batch]:
                continue
            m_pred_bboxes.append(pred_bbox[m_object[batch]].unsqueeze(0))
            m_final_bboxes.append(bboxes_f[batch][m_object[batch]].unsqueeze(0))
            individual_iou = self.iou_2D([m_pred_bboxes[-1]], [m_final_bboxes[-1]])
            individual_iou_var += individual_iou ** 2
        if m_pred_bboxes == []:
            individual_iou = torch.tensor(0, dtype=torch.float32, device=torch.device('cuda'))
        total_iou = self.iou_2D(pred_bboxes, final_bboxes)
        return argO1, argO2, individual_iou, individual_iou_var, total_iou


# Baseline module with splitter
class BaselineModelSplitter(nn.Module):
    
    # Arguments:
    #   vocab: (Vocab)   : Word vocabulary (vocab.json)
    #   configs: (DOView): Baseline configurations (configs.py)
    def __init__(self, vocab, configs):
        super().__init__()
        self.vocab = vocab
        self.configs = configs
        self.single_step = BaselineModel(vocab, configs)
        ################
        ### Splitter ###
        ################
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
        if not configs.model.train_splitter:
            for param in self.embed.parameters():
                param.requires_grad = False
            for param in self.lstm.parameters():
                param.requires_grad = False
            for param in self.linear.parameters():
                param.requires_grad = False
        if configs.model.use_iou_loss:
            self.loss_fn = loss_functions.WeightedSquareIOULoss(box_mode = 'xywh', weights = [100, 100, 100, 100, 90, 50], reduction = None)
        else:
            self.loss_fn = loss_functions.WeightedSquareIOULoss(box_mode = 'xywh', weights = [100, 100, 100, 100, 90, 0], reduction = None)
        self.iou_2D = loss_functions.IOU2D(box_mode = 'xywh', individual_iou = False, reduction = 'sum')

    # Arguments:
    #   batch_dict: (Dictionary): Input data
    # Return Value:
    #   loss: (tensor())          : Loss value
    #   loss_details: (Dictionary): Details of loss (MSE|IoU)
    #   mean_sub_attn: (float)    : Mean correct subject attention value [Only when splitter is pre-trained]
    #   mean_obj_attn: (float)    : Mean correct object attention value  [Only when splitter is pre-trained]
    def forward(self, batch_dict):
        batch_dict = DOView(batch_dict)
        batch_size = batch_dict.initial_image.size(0)
        bboxes_corners = [box[:, 0:4] for box in batch_dict.initial_bboxes]
        f_objects = self.single_step.visual(batch_dict.initial_image, bboxes_corners, batch_dict.object_length)

        e_instructions = self.embed(batch_dict.instruction)
        lengths = batch_dict.instruction_length.cpu()
        p_instructions = RNN.pack_padded_sequence(e_instructions[:, 1:], lengths - 2, batch_first=True, enforce_sorted=False)
        lstm_output = self.lstm(p_instructions)
        linear_output = RNN.PackedSequence(data=self.linear(lstm_output[0].data), batch_sizes=lstm_output[0].batch_sizes, sorted_indices=lstm_output[0].sorted_indices, unsorted_indices=lstm_output[0].unsorted_indices)
        linear_output = RNN.pad_packed_sequence(linear_output, batch_first=True, padding_value=-float('inf'))
        probs = F.softmax(linear_output[0][:, :min(linear_output[0].size(1), 30)], dim=-2)
        
        max_length = batch_dict.instruction.size(1)

        height, width = self.configs.data.image_shape 
        bboxes_i = normalize_bbox(box_convert(batch_dict.initial_bboxes,'xyxy','xywh'), width, height)
        bboxes_f = normalize_bbox(box_convert(batch_dict.final_bboxes,'xyxy','xywh'), width, height)

        pred_bboxes, final_bboxes = [], []
        sub_attn, obj_attn = [], []

        if self.configs.model.train_splitter:
            for batch in range(batch_size):
                # find breakpoint in the natural-language sentence
                init = False
                object_count = f_objects[batch].size(0)
                for bp in range(min(batch_dict.instruction_length[batch] - 2, 30)):
                    instructions = []
                    ins_lengths = []
                    if bp < batch_dict.instruction_length[batch] - 3:
                        # two sentences
                        t_instruction = batch_dict.instruction[batch][:bp + 3].clone()
                        t_instruction[bp + 2] = 2
                        ins_lengths.append(t_instruction.size(0))
                        t_instruction = F.pad(t_instruction, (0, max_length - t_instruction.size(0)))
                        instructions.append(t_instruction)
                        t_instruction = batch_dict.instruction[batch][bp + 1:batch_dict.instruction_length[batch]].clone()
                        t_instruction[0] = 3
                        ins_lengths.append(t_instruction.size(0))
                        t_instruction = F.pad(t_instruction, (0, max_length - t_instruction.size(0)))
                        instructions.append(t_instruction)
                    else:
                        # one sentence
                        t_instruction = batch_dict.instruction[batch][:bp + 3].clone()
                        ins_lengths.append(t_instruction.size(0))
                        t_instruction = F.pad(t_instruction, (0, max_length - t_instruction.size(0)))
                        instructions.append(t_instruction)
                    num_steps = len(instructions)
                    t_bboxes_i = bboxes_i[batch].clone()
                    for t in range(num_steps):
                        f_instruction = self.single_step.language(instructions[t].unsqueeze(0), torch.tensor([ins_lengths[t]], device=torch.device('cuda')))
                        f_action = F.softmax(self.single_step.decoder(f_instruction), dim=-1)

                        subject_instruction = (self.single_step.subject_map(f_instruction)).repeat(object_count, 1)
                        subject_attention = F.softmax(self.single_step.attention_sub(torch.cat([subject_instruction, f_objects[batch]], dim=-1)), dim=0)
                        subject_shape_instruction = (self.single_step.subject_shape_map(f_instruction)).repeat(object_count, 1)
                        subject_shape_attention = F.softmax(self.single_step.attention_shape_sub(torch.cat([subject_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
                        total_subject_attention = subject_attention * subject_shape_attention
                        total_subject_attention = (total_subject_attention) / torch.sum(total_subject_attention, dim=0)

                        object_instruction = (self.single_step.object_map(f_instruction)).repeat(object_count, 1)
                        object_attention = F.softmax(self.single_step.attention_obj(torch.cat([object_instruction, f_objects[batch]], dim=-1)), dim=0)
                        object_shape_instruction = (self.single_step.object_shape_map(f_instruction)).repeat(object_count, 1)
                        object_shape_attention = F.softmax(self.single_step.attention_shape_obj(torch.cat([object_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
                        total_object_attention = object_attention * object_shape_attention
                        total_object_attention = (total_object_attention) / torch.sum(total_object_attention, dim=0)

                        subject_loc = torch.sum(total_subject_attention * t_bboxes_i, dim=0)
                        object_loc = torch.sum(total_object_attention * t_bboxes_i, dim=0)
                        f_embedding = torch.cat([f_action[0], object_loc, subject_loc]).unsqueeze(0)

                        pred = self.single_step.pred(f_embedding)
                        t_bboxes_i = (1 - total_subject_attention) * t_bboxes_i + total_subject_attention * pred
                        if init:
                            pred_bbox = pred_bbox + probs[batch][bp] * t_bboxes_i
                        else:
                            pred_bbox = probs[batch][bp] * t_bboxes_i
                            init = True
                pred_bboxes.append(pred_bbox)
                final_bboxes.append(bboxes_f[batch])
        else:
            for batch in range(batch_size):
                # find breakpoint in the natural-language sentence
                object_count = f_objects[batch].size(0)
                bp = torch.argmax(probs[batch]).item()
                instructions = []
                ins_lengths = []
                if bp < batch_dict.instruction_length[batch] - 3:
                    # two sentences
                    t_instruction = batch_dict.instruction[batch][:bp + 3].clone()
                    t_instruction[bp + 2] = 2
                    ins_lengths.append(t_instruction.size(0))
                    t_instruction = F.pad(t_instruction, (0, max_length - t_instruction.size(0)))
                    instructions.append(t_instruction)
                    t_instruction = batch_dict.instruction[batch][bp + 1:batch_dict.instruction_length[batch]].clone()
                    t_instruction[0] = 3
                    ins_lengths.append(t_instruction.size(0))
                    t_instruction = F.pad(t_instruction, (0, max_length - t_instruction.size(0)))
                    instructions.append(t_instruction)
                else:
                    # one sentence
                    t_instruction = batch_dict.instruction[batch][:bp + 3].clone()
                    ins_lengths.append(t_instruction.size(0))
                    t_instruction = F.pad(t_instruction, (0, max_length - t_instruction.size(0)))
                    instructions.append(t_instruction)
                num_steps = len(instructions)
                t_bboxes_i = bboxes_i[batch].clone()
                subject_attn, object_attn = 0.0, 0.0
                for t in range(num_steps):
                    f_instruction = self.single_step.language(instructions[t].unsqueeze(0), torch.tensor([ins_lengths[t]], device=torch.device('cuda')))
                    f_action = F.softmax(self.single_step.decoder(f_instruction), dim=-1)

                    subject_instruction = (self.single_step.subject_map(f_instruction)).repeat(object_count, 1)
                    subject_attention = F.softmax(self.single_step.attention_sub(torch.cat([subject_instruction, f_objects[batch]], dim=-1)), dim=0)
                    subject_shape_instruction = (self.single_step.subject_shape_map(f_instruction)).repeat(object_count, 1)
                    subject_shape_attention = F.softmax(self.single_step.attention_shape_sub(torch.cat([subject_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
                    total_subject_attention = subject_attention * subject_shape_attention
                    total_subject_attention = (total_subject_attention) / torch.sum(total_subject_attention, dim=0)

                    object_instruction = (self.single_step.object_map(f_instruction)).repeat(object_count, 1)
                    object_attention = F.softmax(self.single_step.attention_obj(torch.cat([object_instruction, f_objects[batch]], dim=-1)), dim=0)
                    object_shape_instruction = (self.single_step.object_shape_map(f_instruction)).repeat(object_count, 1)                    
                    object_shape_attention = F.softmax(self.single_step.attention_shape_obj(torch.cat([object_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
                    total_object_attention = object_attention * object_shape_attention
                    total_object_attention = (total_object_attention) / torch.sum(total_object_attention, dim=0)

                    subject_attn += total_subject_attention[batch_dict.grounded_program[batch][t][1]][0].item()
                    object_attn += total_object_attention[batch_dict.grounded_program[batch][t][2]][0].item()
                    
                    subject_loc = torch.sum(total_subject_attention * t_bboxes_i, dim=0)
                    object_loc = torch.sum(total_object_attention * t_bboxes_i, dim=0)
                    f_embedding = torch.cat([f_action[0], object_loc, subject_loc]).unsqueeze(0)
                    pred = self.single_step.pred(f_embedding)
                    
                    t_bboxes_i = (1 - total_subject_attention) * t_bboxes_i + total_subject_attention * pred
                sub_attn.append(subject_attn / num_steps)
                obj_attn.append(object_attn / num_steps)
                pred_bboxes.append(t_bboxes_i)
                final_bboxes.append(bboxes_f[batch])
        
        losses, loss_details = self.loss_fn(pred_bboxes, final_bboxes)
        loss = torch.stack(losses).mean()
        if not self.configs.model.train_splitter:
            mean_sub_attn = torch.tensor(sub_attn).mean().item()
            mean_obj_attn = torch.tensor(obj_attn).mean().item()
            return loss, loss_details, mean_sub_attn, mean_obj_attn
        else:
            return loss, loss_details
    
    # Arguments:
    #   batch_dict: (Dictionary): Input data
    # Return Value:
    #   argO1: (int)                  : Count of correct subject identification
    #   argO2: (int)                  : Count of correct object identification
    #   individual_iou: (tensor())    : Sum of moved object IoU
    #   individual_iou_var: (tensor()): Sum of squares of moved object IoU
    #   total_iou: (tensor())         : Sum of all object IoU
    def inference(self, batch_dict):
        batch_dict = DOView(batch_dict)
        batch_size = batch_dict.initial_image.size(0)
        bboxes_corners = [box[:, 0:4] for box in batch_dict.initial_bboxes]
        f_objects = self.single_step.visual(batch_dict.initial_image, bboxes_corners, batch_dict.object_length)
        
        max_length = batch_dict.instruction.size(1)

        height, width = self.configs.data.image_shape 
        bboxes_i = normalize_bbox(box_convert(batch_dict.initial_bboxes,'xyxy','xywh'), width, height)
        bboxes_f = normalize_bbox(box_convert(batch_dict.final_bboxes,'xyxy','xywh'), width, height)

        pred_bboxes, final_bboxes = [], []

        argO1, argO2 = 0.0, 0.0
        individual_iou = 0.0
        individual_iou_var = 0.0

        for batch in range(batch_size):
            object_count = f_objects[batch].size(0)
            moved_object_count = len(set([command[1] for command in batch_dict.grounded_program[batch]]))
            instructions = []
            self.decompose(batch_dict.instruction[batch][0:batch_dict.instruction_length[batch]], instructions, max_length)
            
            tempArgO1, tempArgO2 = 0.0, 0.0
            temp_individual_iou = 0.0
            num_steps = len(instructions)
            if num_steps != len(batch_dict.grounded_program[batch]):
                continue
            for t in range(num_steps):
                f_instruction = self.single_step.language(instructions[t].unsqueeze(0), torch.tensor([instructions[t].size(0)], device=torch.device('cuda')))
                f_action = F.softmax(self.single_step.decoder(f_instruction), dim=-1)

                subject_instruction = (self.single_step.subject_map(f_instruction)).repeat(object_count, 1)
                subject_attention = F.softmax(self.single_step.attention_sub(torch.cat([subject_instruction, f_objects[batch]], dim=-1)), dim=0)
                subject_shape_instruction = (self.single_step.subject_shape_map(f_instruction)).repeat(object_count, 1)
                subject_shape_attention = F.softmax(self.single_step.attention_shape_sub(torch.cat([subject_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
                total_subject_attention = subject_attention * subject_shape_attention
                total_subject_attention = (total_subject_attention) / torch.sum(total_subject_attention, dim=0)

                object_instruction = (self.single_step.object_map(f_instruction)).repeat(object_count, 1)
                object_attention = F.softmax(self.single_step.attention_obj(torch.cat([object_instruction, f_objects[batch]], dim=-1)), dim=0)
                object_shape_instruction = (self.single_step.object_shape_map(f_instruction)).repeat(object_count, 1)
                object_shape_attention = F.softmax(self.single_step.attention_shape_obj(torch.cat([object_shape_instruction, f_objects[batch]], dim=-1)), dim=0)
                total_object_attention = object_attention * object_shape_attention
                total_object_attention = (total_object_attention) / torch.sum(total_object_attention, dim=0)

                o1, o2 = torch.argmax(total_subject_attention).item(), torch.argmax(total_object_attention).item()

                if batch_dict.grounded_program[batch][t][1] == o1:
                    tempArgO1 += 1
                if batch_dict.grounded_program[batch][t][2] == o2:
                    tempArgO2 += 1

                subject_loc = bboxes_i[batch][o1]
                object_loc = bboxes_i[batch][o2]
                f_embedding = torch.cat([f_action[0], object_loc, subject_loc]).unsqueeze(0)
                pred = self.single_step.pred(f_embedding)
                bboxes_i[batch][o1] = pred
            argO1 += (tempArgO1 / num_steps)
            argO2 += (tempArgO2 / num_steps)
            pred_bboxes.append(bboxes_i[batch])
            final_bboxes.append(bboxes_f[batch])
            for o in set([command[1] for command in batch_dict.grounded_program[batch]]):
                temp_individual_iou += self.iou_2D([bboxes_i[batch][o].unsqueeze(0)], [bboxes_f[batch][o].unsqueeze(0)])
            temp_individual_iou /= moved_object_count
            individual_iou += temp_individual_iou
            individual_iou_var += (temp_individual_iou) ** 2
        if pred_bboxes == []:
            total_iou = torch.tensor(0, dtype=torch.float32, device=torch.device('cuda'))
        else:
            total_iou = self.iou_2D(pred_bboxes, final_bboxes)
        individual_iou = torch.tensor(individual_iou, dtype=torch.float32, device=torch.device('cuda'))
        individual_iou_var = torch.tensor(individual_iou_var, dtype=torch.float32, device=torch.device('cuda'))
        return argO1, argO2, individual_iou, individual_iou_var, total_iou
        
    # Arguments:
    #   instruction: (tensor(L))    : Given instruction to split
    #   instructions: (list(tensor)): List of atomic instructions
    #   max_length: (int)           : Maximum length of instruction in batch
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
