import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F


from datasets.roboclevr.instruction_transforms import encode_using_lexed_sentence
from datasets.common.collate import SimpleCollate
from datasets.definition import gdef
from helpers.utils.container import DOView
from datasets.common.program_translator import nsrmseq_to_nsrmtree, append_nsrmtrees, nsrmtree_to_nsrmseq

import jactorch.nn as jacnn
from helpers.logging import get_logger
from copy import deepcopy 
import torch.nn.utils.rnn as  RNN 

logger  = get_logger(__file__)

class SentenceGrainer(nn.Module):
    def __init__(self,vocab,  hidden_dim, word_emb_dim, max_step,positional_emb_dim=None,device='cuda'):
        super().__init__()
        self.device = device 
        self.max_step = max_step
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding_dim  = word_emb_dim
        self.positional_embedding_dim = positional_emb_dim
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim, batch_first=True,dropout=0.5)
        self.linear = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self,sent,sent_length,exhaustive_search=True):


        if self.training and exhaustive_search :
            #asssume at most one split 
            e_instructions = self.embedding(sent)
            sent_length = sent_length.cpu() ##
            p_instructions = RNN.pack_padded_sequence(e_instructions[:, 1:], sent_length - 2, batch_first=True, enforce_sorted=False)
            lstm_output = self.lstm(p_instructions)
            linear_output = RNN.PackedSequence(data=self.linear(lstm_output[0].data), batch_sizes=lstm_output[0].batch_sizes, sorted_indices=lstm_output[0].sorted_indices, unsorted_indices=lstm_output[0].unsorted_indices)
            linear_output = RNN.pad_packed_sequence(linear_output, batch_first=True, padding_value=-float('inf'))
            probs = F.softmax(linear_output[0][:, :min(linear_output[0].size(1), 30)], dim=-2)
            max_len = max(sent_length)
            all_instructions = []
            all_instructions_lengths = []
            for batch in range(len(sent)):
                possible_pairs = []
                possible_pair_lens = []
                for bp in range(min(sent_length[batch] - 2, 30)):
                    instructions = []
                    lens = []
                    if bp < sent_length[batch] - 3:
                        # two sentences
                        t_instruction = sent[batch][:bp + 2].clone()
                        t_instruction[bp + 1] = 2
                        lens.append(t_instruction.size(0))
                        t_instruction = F.pad(t_instruction, (0, max_len - t_instruction.size(0)))
                        instructions.append(t_instruction)
                        t_instruction = sent[batch][bp + 1:sent_length[batch]].clone()
                        t_instruction[0] = 3
                        lens.append(t_instruction.size(0))
                        t_instruction = F.pad(t_instruction, (0, max_len - t_instruction.size(0)))
                        instructions.append(t_instruction)
                    
                    else:
                        # one sentence
                        t_instruction = sent[batch][:bp + 3].clone()
                        lens.append(t_instruction.size(0))
                        t_instruction = F.pad(t_instruction, (0, max_len - t_instruction.size(0)))
                        instructions.append(t_instruction)

                    possible_pairs.append(instructions)
                    possible_pair_lens.append(lens)

                all_instructions.append(possible_pairs)
                all_instructions_lengths.append(possible_pair_lens)

            return all_instructions, all_instructions_lengths, probs 
        

        else: 
            max_len = max(sent_length)
            all_instructions = []
            all_instructions_lengths = []
            for i in range(sent.size(0)):
                splitted_instructions = []
                splitted_lens = []
                self.decompose(sent[i][:sent_length[i]],splitted_instructions,splitted_lens,max_len)
                all_instructions.append([splitted_instructions])
                all_instructions_lengths.append([splitted_lens])

            return all_instructions, all_instructions_lengths, None 

    def decompose(self, instruction, instructions, lens, max_length):
        if instruction.size(0) < 3:
            return 
        instruction = instruction.clone()
        x = instruction.size(0)
        length = torch.tensor([instruction.size(0)], device=torch.device('cpu'))
        instruction = F.pad(instruction, (0,max_length - instruction.size(0)))
        e_instruction = self.embedding(instruction.unsqueeze(0))
        p_instructions = RNN.pack_padded_sequence(e_instruction[:, 1:], length - 2, batch_first=True, enforce_sorted=False)
        lstm_output = self.lstm(p_instructions)
        linear_output = RNN.PackedSequence(data=self.linear(lstm_output[0].data), batch_sizes=lstm_output[0].batch_sizes, sorted_indices=lstm_output[0].sorted_indices, unsorted_indices=lstm_output[0].unsorted_indices)
        linear_output = RNN.pad_packed_sequence(linear_output, batch_first=True, padding_value=-float('inf'))
        probs = F.softmax(linear_output[0][0,:min(linear_output[0].size(1),30)], dim=-2) # assuming one instruction occurs within 30 wors 
        bp = torch.argmax(probs).item()
        if bp < length - 3:
            t_instruction = instruction[:bp + 2].clone()
            t_instruction[bp + 1] = 2
            self.decompose(t_instruction, instructions, lens, max_length)
            s_instruction = instruction[bp + 1:length].clone()
            s_instruction[0] = 3
            self.decompose(s_instruction, instructions, lens, max_length)
        else:
            t_instruction = instruction.clone()
            instructions.append(t_instruction)
            lens.append(x)


class ProgramBuilder(nn.Module):
    def __init__(self, program_parser,vocab,  splitter_hidden_dim,splitter_word_emb_dim, max_step,   splitter_gru_nlayers = 1 , use_bert = False, splitter_pos_emb_dim=None,penalty=1):
        super().__init__()
        self.splitter  = SentenceGrainer(vocab, splitter_hidden_dim, splitter_word_emb_dim, max_step, use_bert,  splitter_pos_emb_dim)
        self.parser = program_parser
        self.max_step = max_step
        self.penalty = penalty 
        self.vocab = vocab 
        self.collate = SimpleCollate({'instruction':'pad', 'mask':'pad'})

    def check_syntactic_validity(self,sent):
        return True 

    def forward(self, sent, sent_length, concept_groups, relational_concept_groups, action_concept_groups, parser_max_depth):

        broken_sent,broken_sent_lengths, masks_prob  = self.splitter(sent, sent_length)
        all_sent , all_sent_lengths, all_cgs,all_rcgs,all_ags = [],[],[],[],[]
        assert sent_length.size(0) ==  len(broken_sent) 
        for i in range(sent_length.size(0)):

            for j,(parts,parts_lengths) in enumerate(zip(broken_sent[i],broken_sent_lengths[i])):   
                seen_cg,seen_ag,seen_rcg = 0,0,0

                for p,p_length in zip(parts,parts_lengths):
                    this_sent = p
                    this_length = p_length 
                    num_cg,num_rcg,num_ag = (torch.tensor(p) == self.vocab.word2idx["<CONCEPTS>"]).sum(),(torch.tensor(p) == self.vocab.word2idx["<REL_CONCEPTS>"]).sum(),(torch.tensor(p)== self.vocab.word2idx["<ACT_CONCEPTS>"]).sum()
                    cg,rcg,ag = concept_groups[i][seen_cg : seen_cg+num_cg],relational_concept_groups[i][seen_rcg:seen_rcg+num_rcg], action_concept_groups[i][seen_ag:seen_ag+num_ag]
                    seen_cg,seen_rcg,seen_ag = seen_cg+num_cg,seen_rcg+num_rcg,seen_ag+num_ag 
                    all_sent.append(this_sent)
                    all_sent_lengths.append(this_length)
                    all_cgs.append(cg)
                    all_rcgs.append(rcg)
                    all_ags.append(ag) 

        all_sent,all_sent_lengths = torch.stack(all_sent,dim=0).to(sent.device),torch.tensor(all_sent_lengths,device=sent.device)
        all_sent = all_sent[:,:max(all_sent_lengths)]
        if self.parser.use_bert:
            raise AttributeError("Splitter cannot be used with bert in parser. Please disable bert in parser")
        try:
            all_programs =  self.parser(all_sent, all_sent_lengths, parser_max_depth, 
                            all_cgs,all_rcgs,all_ags)

        except Exception as e:
            logger.error(f'\nError in parser execution', exc_info = True, stack_info = True)
            exit()

        programs = []
        program_idx = 0
        is_minimal = []
        for idx,sent_pairs in enumerate(broken_sent):
            this_minimal = []
            for j,_ in enumerate(sent_pairs):
                composed_programs = all_programs[program_idx:program_idx+len(broken_sent[idx][j])]
                
                program_idx += len(broken_sent[idx][j])

                prog_trees = [nsrmseq_to_nsrmtree(p['program']) for p in composed_programs]
                
                minimal =  True 
                for prog_tree in prog_trees:
                    if prog_tree['op'] == 'idle':
                        minimal = False 
                        break 
                this_minimal.append(minimal)

                if self.training:
                    prog_desc = dict(scene_id = idx, program = nsrmtree_to_nsrmseq(append_nsrmtrees(prog_trees)),
                     log_likelihood = masks_prob[idx][j][0], discounted_log_likelihood = masks_prob[idx][j][0])
                else: 
                    prog_desc = dict(scene_id = idx, program = nsrmtree_to_nsrmseq(append_nsrmtrees(prog_trees)),
                     log_likelihood = torch.tensor(1.0,device=sent.device), discounted_log_likelihood = torch.tensor(1.0,device=sent.device))  

                programs.append(prog_desc)
            
            is_minimal.append(this_minimal)
        num_pairs = [len(x) for x in broken_sent]
        return programs,num_pairs,is_minimal  