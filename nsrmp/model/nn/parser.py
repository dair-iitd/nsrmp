import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import jactorch
import jactorch.nn as jacnn

from jacinle.utils.enum import JacEnum
from datasets.definition import gdef
from datasets.common.program_translator import nsrmseq_to_nsrmqsseq, nsrmtree_to_nsrmseq
from .parser_features import QueueExecutor

__all__ = ['SampleMethod','ProgramParser']

class SampleMethod(JacEnum):
    GROUNDTRUTH = 'groundtruth'
    SAMPLE = 'sample'
    SAMPLE_UNIFORM = 'sample-uniform'
    SAMPLE_ENUMERATE = 'sample-enumerate'

class ProgramParser(nn.Module):
    """The final program parser."""

    def __init__(self, vocab, gru_hidden_dim, gru_nlayers, word_emb_dim, positional_emb_dim, use_bert = False,  discount=None):

        """
        Args:
            vocab: The vocab set.
            hidden_dim (int): the hidden dimension for the GRU encoder.
            discount (float): the discount factor for the computing discounted returns.
        """

        super().__init__()
        self.use_bert = use_bert
        self.discount = discount
        self.operations = gdef.operation_signatures
        self.operation2idx = {k: i for i, (k, _, _, _) in enumerate(self.operations)}
        
        for x in gdef.parameter_types:
            # NB(Jiayuan Mao @ 10/03): we assume each operator has at most one parameter..
            mask = [1 if len(v) > 0 and v[0] == x  else 0 for _, v, _, _ in self.operations]
            self.register_buffer('operations_mask_p_' + x, torch.tensor(mask, dtype=torch.float))

        for x, y in zip(('variable', 'return'), (gdef.variable_types, gdef.return_types)):
            mask = [1 if v in y else 0 for _, _, _, v in self.operations]
            self.register_buffer('operations_mask_o_' + x, torch.tensor(mask, dtype=torch.float))
            
        mask = [1 if len(v) == 0 else 0 for _, _, v, _ in self.operations]
        self.register_buffer('operations_mask_i_none', torch.tensor(mask, dtype=torch.float))

        if self.use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", framework = 'pt', additional_special_tokens = gdef.extra_embeddings, device = 0)
            self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            self.bert.resize_token_embeddings(len(self.tokenizer))
            self.hidden_dim = 64
        else:
            self.vocab = vocab
            self.hidden_dim = gru_hidden_dim
            self.embedding_dim  = word_emb_dim
            self.positional_embedding_dim = positional_emb_dim
            self.tot_embedding_dim = self.embedding_dim 
            self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
            if self.positional_embedding_dim is not None:
                from jactorch.nn.embedding import LearnedPositionalEmbedding
                self.positional_embedding = LearnedPositionalEmbedding(128,self.positional_embedding_dim)
                self.tot_embedding_dim += self.positional_embedding_dim
            else:
                self.add_module('positional_embedding', None)
            self.gru = jacnn.GRULayer(self.tot_embedding_dim, self.hidden_dim, gru_nlayers, bidirectional=True, batch_first=True, dropout=0.5)
        self.queue = QueueExecutor(2*self.hidden_dim, len(self.operations))
        self.concept_attentions = nn.ModuleDict()
        for k in gdef.parameter_types:
            self.concept_attentions[k] = nn.Sequential(nn.Linear(2*(2*self.hidden_dim),2*self.hidden_dim), nn.ReLU(), nn.Linear(2*self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim,1))
        
    def encode_sentence(self, sent, sent_length):
        if self.use_bert:
            bert_inputs = self.tokenizer(sent,  return_tensors = 'pt', padding = True).to(self.bert.device)
            outputs = self.bert(**bert_inputs)
            return bert_inputs['input_ids'], outputs.last_hidden_state, outputs.last_hidden_state[:,-1,:]
        else:
            f = self.embedding(sent)
            if self.positional_embedding is not None:
                f = torch.cat([f, self.positional_embedding(sent)], dim=-1)
            hiddens, last_state = self.gru(f, sent_length)
            return sent, hiddens, last_state
    
    def decode_sentence(self, sent):
        if self.use_bert:
            raise NotImplementedError
        else:
            return [self.vocab.idx2word[t.item()] for t in sent]
    
    def _token2ids(self,str_token):
        assert type(str_token) == str
        if self.use_bert:
            return self.tokenizer.convert_tokens_to_ids(str_token)
        else:
             return self.vocab.word2idx[str_token]

    def forward(self,sent,sent_length,max_depth,concept_groups,relational_concept_groups=[],action_concept_groups=[],sample_method='sample',sample_space=None, exploration_rate = 0, mask = None):
        sample_method = SampleMethod.from_string(sample_method)
        if sample_method in (SampleMethod.SAMPLE, SampleMethod.SAMPLE_UNIFORM):
            assert sample_space is None, 'Sample with sample space support is not implemented.'
        if not self.training:
            sample_method, sample_space = SampleMethod.SAMPLE, None

        programs = list()
        sent,hiddens,last_state = self.encode_sentence(sent,sent_length)

        for i, l in enumerate(sent_length):
            l = l.item()
            if mask is not None:
                s, h, f = mask[i, :l]*sent[i, :l], mask[i, :l].unsqueeze(-1)*hiddens[i, :l], last_state[i]
            else:
                s, h, f = sent[i], hiddens[i,:], last_state[i]

            cg_mask = (s == self._token2ids("<CONCEPTS>"))

            assert cg_mask.sum().item() == len(concept_groups[i]), print(concept_groups[i],i,sent[i],sent_length[i])
            cgs = h[cg_mask]
            rcg_mask = (s == self._token2ids('<REL_CONCEPTS>'))
            assert rcg_mask.sum().item() == len(relational_concept_groups[i])
            rcgs = h[rcg_mask]
            ag_mask = (s == self._token2ids('<ACT_CONCEPTS>'))
            assert ag_mask.sum().item() == len(action_concept_groups[i])
            ags = h[ag_mask]
            info = dict(
                encoder_embeddings = hiddens[i],
                parameter_masks = dict(attribute_concept = cg_mask, relational_concept = rcg_mask, action_concept = ag_mask),
                parameter_embeddings=dict(
                    attribute_concept=cgs, relational_concept=rcgs, action_concept=ags
                ), parameter_values=dict(
                    attribute_concept=concept_groups[i],
                    relational_concept=relational_concept_groups[i],
                    action_concept=action_concept_groups[i]
                ), ret_type='return'
            )

            if sample_method in (SampleMethod.SAMPLE, SampleMethod.SAMPLE_UNIFORM):
                info['p'], info['lp'] = dict(), list()
                programs.append((i, info['p'], info['lp']))  # scene_id, program, log_prob
                self.queue.enqueue(info, last_state=f, last_operation=0, param_id=0)
            elif sample_method in (SampleMethod.GROUNDTRUTH, SampleMethod.SAMPLE_ENUMERATE):
                this_sample_space = sample_space[i]
                if len(this_sample_space) == 0:
                    info['p'], info['lp'] = dict(), list()
                    programs.append((i, info['p'], info['lp']))  # scene_id, program, log_prob
                    self.queue.enqueue(info, last_state=f, last_operation=0, param_id=0)
                else:
                    if sample_method is SampleMethod.GROUNDTRUTH:
                        this_sample_space = [this_sample_space]
                    for instance in this_sample_space:
                        this_info = info.copy()
                        this_info['p'], this_info['lp'] = dict(), list()
                        this_info['sample_space'] = instance
                        programs.append((i, this_info['p'], this_info['lp']))  # scene_id, program, log_prob
                        self.queue.enqueue(this_info, last_state=f, last_operation=0, param_id=0)
            else:
                raise ValueError('Unknown sample method: {}.'.format(sample_method))
    
        for current_depth in range(max_depth):
            if self.queue.empty():
                break
            infos, inputs, outputs = self.queue.execute()
            for i, info in enumerate(infos):
                p, lp, pembs = info['p'], info['lp'], info['parameter_embeddings']
                assert i < outputs['operation'].size(0)

                # We have two masks: one for syntactically correctness, and one for constraint sampling.
                syntax_mask, sample_mask = self._get_operation_sample_mask(
                    info['ret_type'], current_depth, max_depth,
                    pembs['attribute_concept'].size(0), pembs['relational_concept'].size(0), pembs['action_concept'].size(0),
                    info.get('sample_space', None)
                )

                operation_logits = outputs['operation'][i] * syntax_mask + (-1e9) * (1 - syntax_mask)
                op_idx, log_prob = self._sample(operation_logits, sample_method, sample_mask=sample_mask, epsilon = exploration_rate)
                op_info = self.operations[op_idx]
                p['op'] = op_info[0]; lp.append(log_prob)

                # Use attention score to choose the parameter (concept, relational_concept, attribute).
                parameter_type = None
                if len(op_info[1]) != 0:
                    assert len(op_info[1]) == 1
                    parameter_type = op_info[1][0]

                if parameter_type is None:
                    pass
                else:
                    parameter_sample = None
                    if 'sample_space' in info:
                        parameter_sample = info['sample_space'].get(parameter_type + '_idx', None)
                    p[parameter_type + '_idx'], log_prob = self._get_parameter_attention(parameter_type,
                         outputs[parameter_type][i], outputs['attentions'][i][info['parameter_masks'][parameter_type]], info['parameter_embeddings'][parameter_type],
                        parameter_sample, exploration_rate = exploration_rate)
                    p[parameter_type + '_values'] = info['parameter_values'][parameter_type]
                    p['param_type'] = parameter_type
                    # p[parameter_type] = p[parameter_type + '_values'][p[parameter_type + '_idx']]
                    lp.append(log_prob)
                    
                # Determine the inputs by BFS.
                input_types = op_info[2]
                p['inputs'] = []
                for param_id, input_type in enumerate(input_types):
                    param = dict()
                    if input_type == 'object_set':
                        p['inputs'].append(param)
                    elif input_type == 'object':
                        p['inputs'].append(param)
                    elif input_type == 'world':
                        p['inputs'].append(param)
                    else:
                        raise ValueError('Unknown input type: {}.'.format(input_type))

                    next_info = info.copy()
                    next_info['p'] = param
                    # TODO(Jiayuan Mao @ 10/07): this should be a fine-grained variable type.
                    next_info['ret_type'] = 'variable'

                    #NSRMP_begin
                    if param_id == 2:
                        next_info['ret_type'] = 'return'
                    #NSRMP_end (This change does make the parser dependent on the DSL, need to be discussed)

                    if 'sample_space' in info:
                        next_info['sample_space'] = info['sample_space']['inputs'][param_id]
                    self.queue.enqueue(
                        next_info,
                        last_state=outputs['current_state'][i], last_operation=op_idx, param_id=param_id + 1,
                        # last_context_vector = outputs["context_vector"][i],
                    )
        self.queue.check_finalized()
        x = self._translate(programs)
        return list(x)

    def _get_operation_sample_mask(self, ret_type, current_depth, max_depth, nr_cgs, nr_rcgs, nr_ags, sample_space):
        # Filter the return type.
        if ret_type == 'return':
            syntax_mask = self.operations_mask_o_return
        elif ret_type == 'variable':
            syntax_mask = self.operations_mask_o_variable
        else:
            raise ValueError('Unknown type in parser: {}.'.format(ret_type))

        syntax_mask = syntax_mask.clone().detach()

        # Filter the depth.
        if current_depth == max_depth - 1:
            syntax_mask.mul_(self.operations_mask_i_none)
        for nr, mask in zip([nr_cgs, nr_rcgs, nr_ags], ['attribute_concept', 'relational_concept', 'action_concept']):
            if nr == 0:
                syntax_mask.mul_(1 - getattr(self, 'operations_mask_p_' + mask))

        # Filter the given search space.
        # NB(Jiayuan Mao): Assume that there is only one operation.
        sample_mask = None
        if sample_space is not None:
            sample_mask = self._one_hot(self.operation2idx[sample_space['op']], syntax_mask)

        return syntax_mask, sample_mask

    def _get_parameter_attention(self,parameter_type, att_key,att_scores, att_values, idx=None, exploration_rate = 0):
        attention = self.concept_attentions[parameter_type](torch.cat([ att_key.unsqueeze(0).expand(att_values.shape),att_scores.unsqueeze(-1)*att_values], dim =-1)).squeeze(-1)
        sample_mask = None
        if idx is not None:
            sample_mask = self._one_hot(idx, attention)
        return self._sample(attention, SampleMethod.SAMPLE, sample_mask, epsilon= exploration_rate)

    def _sample(self, logits, sample_method, sample_mask=None,epsilon = 0):
        log_probs = F.log_softmax(logits, dim=-1)
        if sample_mask is not None:
            logits = logits * sample_mask + (-1e9) * (1 - sample_mask)
        probs = F.softmax(logits, dim=-1)
        if self.training:
            if sample_method is SampleMethod.SAMPLE:
                rnd = random.random()
                if rnd <epsilon:
                    val = torch.multinomial((logits > -1e8).float(), 1).item()
                else:
                    val = torch.multinomial(probs, 1).item()
            elif sample_method in (SampleMethod.SAMPLE_UNIFORM, SampleMethod.SAMPLE_ENUMERATE, SampleMethod.GROUNDTRUTH):
                val = torch.multinomial((logits > -1e8).float(), 1).item()  # uniformly sample from the possible operations.
            else:
                raise ValueError('Unknown sample method: {}.'.format(sample_method))
        else:
            val = probs.argmax(dim=0).item()
        return val, log_probs[val]

    def _translate(self, programs):
    
        x = []
        for scene_id, prog, lps in programs:
            # let's hope that Torch will optimize the back-propagation.
            lps = torch.stack(lps, dim=0)
            if self.discount is None:
                likelihood, discounted_likelihood = lps.sum(), lps.sum()
            else:
                likelihood = lps.sum()
                dis = torch.arange(lps.size(0) - 1, -1, -1, dtype=torch.float, device=lps.device)
                dis = self.discount ** dis
                discounted_likelihood = (lps * dis).sum()

            x.append({
                'scene_id': scene_id, 'program': nsrmtree_to_nsrmseq(prog),
                'log_likelihood': likelihood,
                'discounted_log_likelihood': discounted_likelihood
            })

        return x

    @classmethod
    def _one_hot(cls, idx, tensor):
        return jactorch.one_hot(torch.tensor([idx], dtype=torch.long, device=tensor.device), tensor.size(0))[0]

    def reset_parameters(self):
        for layer in self.gru.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.queue.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()