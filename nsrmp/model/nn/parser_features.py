import collections
from model import configs
import torch
import torch.nn as nn

import torch.nn.functional as F 
import jactorch
import jactorch.nn as jacnn


class BatchExecutor(nn.Module):
    def __init__(self,hidden_dim,nr_operations):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nr_operations = nr_operations
        self.param1_attention = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(), nn.Linear(hidden_dim,1))
        self.param2_attention = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(), nn.Linear(hidden_dim,1))
        self.param3_attention = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Tanh(), nn.Dropout(), nn.Linear(hidden_dim,1))

        self.context_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(), nn.ReLU(), nn.Linear(hidden_dim,hidden_dim))
        self.action_concept_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(), nn.ReLU(),   nn.Linear(hidden_dim,hidden_dim))
        self.operation_context_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(), nn.ReLU(),  nn.Linear(hidden_dim,hidden_dim))
        self.relational_concept_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(), nn.ReLU(),  nn.Linear(hidden_dim,hidden_dim))
        self.attribute_concept_fc  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(), nn.ReLU(),  nn.Linear(hidden_dim,hidden_dim))

        self.operation_classifier = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),nn.Dropout(p=0.3), nn.Linear(self.hidden_dim,self.nr_operations))
        self.concept_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(self.hidden_dim,self.hidden_dim))
        self.relational_concept_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),nn.Dropout(p=0.3), nn.Linear(self.hidden_dim,self.hidden_dim))
        self.action_concept_attention = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(self.hidden_dim,self.hidden_dim))
      
        self.param1_decoder = nn.GRUCell(self.hidden_dim + self.nr_operations, self.hidden_dim)
        self.param2_decoder = nn.GRUCell(self.hidden_dim + self.nr_operations, self.hidden_dim)
        self.param3_decoder = nn.GRUCell(self.hidden_dim + self.nr_operations, self.hidden_dim)

        self.input_spec = dict(
            last_state='default',
            last_operation='python_int',
            param_id='python_int',
            last_context_vector = 'default'
        )

        self.output_spec = dict(
            current_state='default',
            operation='default',
            attribute_concept='default',
            relational_concept='default',
            action_concept='default',
            context_vector = 'default',
            attentions = "default",
        )
    
    def forward(self,hidden_states,last_state,last_operation, param_id, last_context_vector = None, batch_process = True):
        
        '''
        Args:
            last_state: shape `[batch_size, hidden_dim]`.
            last_operation: shape `[batch_size, nr_operations]` or `[batch_size]`.
            param_id: shape `[batch_size]`
            (param_id is the index in the operation_signatures 
        corresponding to which we are trying to decode)
        Returns:
            output_dict (dict): see `self.output_spec`.
        '''
        

        if (param_id == 0).any().item():
            #sanity check, all are initialised(just have been put into the queue)
            assert (param_id == 0).all().item()
            current_state = last_state
            attention = torch.ones(current_state.size(0), hidden_states.size(1)).to(device = hidden_states.device)
        else:
            
            param_id = param_id.to(last_state.device)
            last_operation = last_operation.to(last_state.device)
            if last_operation.dim() == 1:
                last_operation = jactorch.one_hot(last_operation, self.nr_operations)
            # last_operation = torch.cat([last_operation,last_context_vector],dim=-1)

            '''
            since different examples in one batch maye correspond to 
            the second or first(and for actions, third) argument,
            we need to apply different decoders to them.To di this smoothly
            in batches, the code block below stitches them accordingly
            '''
            
            if (param_id == 1).all().item():
                param1_attended_scores = self.param1_attention(torch.cat([last_state.unsqueeze(1).expand(hidden_states.shape),hidden_states], dim = -1))
                param1_attended_scores = F.softmax(param1_attended_scores,dim = -2)
                param1_context_vector = torch.bmm(hidden_states.permute(0,2,1),param1_attended_scores).squeeze(-1)
                current_state = self.param1_decoder(torch.cat([last_operation, param1_context_vector],dim=-1), last_state)
                attention = param1_attended_scores.squeeze(-1)
            else:
                batch_size = last_state.size(0)
                sorted_param_id,perm_idx = param_id.sort(0,descending=False)
                last_state = last_state[perm_idx]
                last_operation = last_operation[perm_idx]
                hidden_states = hidden_states[perm_idx]
                #to do 
                split_12 = torch.arange(0, batch_size, dtype=torch.int64, device=last_state.device)[sorted_param_id.eq(1)].max().item() + 1
                split_23 = torch.arange(0, batch_size, dtype=torch.int64, device=last_state.device)[sorted_param_id.eq(2)].max().item() + 1
                current_states = []
                attentions = []
                
                if split_12 > 0:
                    this_hidden_states = hidden_states[:split_12]
                    param1_attended_scores = self.param1_attention(torch.cat([last_state[:split_12].unsqueeze(1).expand(this_hidden_states.shape),this_hidden_states], dim = -1))
                    param1_attended_scores = F.softmax(param1_attended_scores,dim = -2)
                    param1_context_vector = torch.bmm(this_hidden_states.permute(0,2,1),param1_attended_scores).squeeze(-1)
                    current_states.append(self.param1_decoder(torch.cat([last_operation[:split_12],param1_context_vector], dim = -1), last_state[:split_12]))
                    attentions.append(param1_attended_scores.squeeze(-1))
                if split_12 < split_23 :
                    this_hidden_states = hidden_states[split_12:split_23]
                    param2_attended_scores = self.param2_attention(torch.cat([last_state[split_12:split_23].unsqueeze(1).expand(this_hidden_states.shape),this_hidden_states], dim = -1))
                    param2_attended_scores = F.softmax(param2_attended_scores,dim = -2)
                    param2_context_vector = torch.bmm(this_hidden_states.permute(0,2,1),param2_attended_scores).squeeze(-1)
                    current_states.append(self.param2_decoder(torch.cat([last_operation[split_12:split_23],param2_context_vector], dim = -1), last_state[split_12:split_23]))
                    attentions.append(param2_attended_scores.squeeze(-1))
                if split_23< batch_size:
                    this_hidden_states = hidden_states[split_23:]
                    param3_attended_scores = self.param3_attention(torch.cat([last_state[split_23:].unsqueeze(1).expand(this_hidden_states.shape),this_hidden_states], dim = -1))
                    param3_attended_scores = F.softmax(param3_attended_scores,dim = -2)
                    param3_context_vector = torch.bmm(this_hidden_states.permute(0,2,1),param3_attended_scores).squeeze(-1)
                    current_states.append(self.param3_decoder(torch.cat([last_operation[split_23:],param3_context_vector], dim = -1), last_state[split_23:]))
                    attentions.append(param3_attended_scores.squeeze(-1))
                
                if len(current_states) == 1:
                    current_state = current_states[0]
                    attention = attentions[0]
                else:
                    current_state = torch.cat(current_states, 0)
                    attention = torch.cat(attentions,0)

                inverse_perm_idx = jactorch.inverse_permutation(perm_idx)
                current_state = current_state.index_select(0, inverse_perm_idx)
                attention = attention.index_select(0,inverse_perm_idx)

        # attended_scores = (current_state.unsqueeze(1) * hidden_states).sum(dim=-1, keepdims = True)
        # attended_scores = F.softmax(attended_scores,dim = -2)
        # context_vector = self.context_fc(torch.bmm(hidden_states.permute(0,2,1),attended_scores).squeeze(-1))
        
        # operation = self.operation_classifier(torch.tanh(self.operation_context_fc(current_state)+context_vector))
        # concept = self.concept_attention(torch.tanh(self.attribute_concept_fc(current_state)+ context_vector))
        # action_concept = self.action_concept_attention(torch.tanh(self.action_concept_fc(current_state)+ context_vector))
        # relational_concept = self.relational_concept_attention(torch.tanh(self.relational_concept_fc(current_state)+context_vector))

        # operation = self.operation_classifier(torch.cat([self.operation_context_fc(current_state),context_vector], dim =-1))
        # concept = self.concept_attention(torch.cat([self.attribute_concept_fc(current_state), context_vector], dim =-1))
        # action_concept = self.action_concept_attention(torch.cat([self.action_concept_fc(current_state), context_vector], dim = -1))
        # relational_concept = self.relational_concept_attention(torch.cat([self.relational_concept_fc(current_state) ,context_vector] , dim = -1))
     
        operation = self.operation_classifier(current_state)
        concept = self.concept_attention(current_state)
        action_concept = self.action_concept_attention(current_state)
        relational_concept = self.relational_concept_attention(current_state)

        return dict(current_state=current_state, operation=operation, attribute_concept=concept, relational_concept = relational_concept, action_concept = action_concept, attentions = attention)


class QueueExecutor(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_executor = BatchExecutor(*args, **kwargs)
        self.info_queue = list()
        self.queue = collections.defaultdict(list)

    ###add input/output spec
    def enqueue(self, info, **kwargs):
        self.info_queue.append(info)
        for k, v in kwargs.items():
            self.queue[k].append(v)
    
    def execute(self):
        inputs_info, inputs = self.info_queue.copy(), self.queue.copy()
        self.info_queue.clear()
        self.queue.clear()

        stacked_input = dict()
        stacked_input['hidden_states'] = torch.stack([info['encoder_embeddings'] for info in inputs_info])
        for k, v in inputs.items():
            assert k in self.input_spec, 'Unspecified input: {}.'.format(k)
            if self.input_spec[k] == 'default':
                stacked_input[k] = torch.stack(v, dim=0)
            elif self.input_spec[k] == 'python_int':
                stacked_input[k] = torch.tensor(v, dtype=torch.int64)
            else:
                raise ValueError('Unknown input spec for {}: {}.'.format(k, self.input_spec[k]))
        
        outputs = self.batch_executor(**stacked_input)
        return inputs_info, inputs, outputs

    @property
    def input_spec(self):
        return self.batch_executor.input_spec

    @property
    def output_spec(self):
        return self.batch_executor.output_spec

    def empty(self):
        return len(self.info_queue) == 0

    def check_finalized(self):
        # NB(Jiayuan Mao): self.queue should also be "0" XD.
        assert len(self.queue) == len(self.info_queue) == 0
