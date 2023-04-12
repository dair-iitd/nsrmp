#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   instruction_transforms.py
#Time    :   2022/06/10 21:02:46
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

from datasets.definition import gdef
import random 

def extract_concepts(sent, group_concepts, gdef):
    #Bug: This can group only one concept type. If group_concepts is a list of len>=2, then this will fail
    concept_groups = gdef.concept_groups
    output_dict = {k:list() for k in concept_groups}
    concept_words_dict = gdef.concept_words_dict
    all_concept_words = gdef.all_concept_words
    output_tokens = []
    last_concept_group = None
    for w in sent:
        if w in all_concept_words:
            for concept_type in concept_groups:
                this_concept_words = concept_words_dict[concept_type]
                if w in this_concept_words:
                    if group_concepts is None or concept_type not in group_concepts:
                        output_dict[concept_type].append([w])
                        output_tokens.append(gdef.con2ebd[concept_type])
                        break
                    elif  concept_type in group_concepts:
                        if last_concept_group is None:
                            last_concept_group = [w]
                            output_dict[concept_type].append(last_concept_group)
                            output_tokens.append(gdef.con2ebd[concept_type])
                        else:
                            last_concept_group.append(w)
                       
                        break
        else:
            output_tokens.append(w)
            last_concept_group = None
    return output_tokens, output_dict



def extract_lexed_concepts(sent, sent_lexed, group_concepts, gdef):
    #Bug: This can group only one concept type. If group_concepts is a list of len>=2, then this will fail
    concept_groups = gdef.concept_groups
    output_dict = {k:list() for k in concept_groups}
    all_concept_words = gdef.all_concept_words
    concept_tokens = gdef.concept_tokens(sent_lexed)
    output_tokens = []
    last_concept_group = None
    for w in sent:
        if w in all_concept_words:
            if len(concept_tokens) <= 0:
                breakpoint()
            assert len(concept_tokens) > 0 , "Error in lexing. Debug extract_lexed_concepts"
            concept_type = concept_tokens.pop(0)
            if group_concepts is None or concept_type not in group_concepts:
                output_dict[concept_type].append([w])
                output_tokens.append(gdef.con2ebd[concept_type])
            elif  concept_type in group_concepts:
                if last_concept_group is None:
                    last_concept_group = [w]
                    output_dict[concept_type].append(last_concept_group)
                    output_tokens.append(gdef.con2ebd[concept_type])
                else:
                    last_concept_group.append(w)
        else:
            output_tokens.append(w)
            last_concept_group = None
    return output_tokens, output_dict

def replace_by_synonyms(sent,word2lemma):
    for idx, w in enumerate(sent):
        if w in word2lemma.keys():
            sent[idx] = word2lemma[w]
    return sent

def encode_sentence(sent:list, group_concepts:list):
    sent_new = replace_by_synonyms(sent,gdef.word2lemma)
    tokens,concept_dict =  extract_concepts(sent_new, group_concepts, gdef)
    return tokens, concept_dict

def encode_using_lexed_sentence(sent:list, sent_lexed:list, group_concepts:list):
    sent_new = replace_by_synonyms(sent,gdef.word2lemma)
    tokens,concept_dict =  extract_lexed_concepts(sent_new, sent_lexed, group_concepts, gdef)
    return tokens, concept_dict





def join_single_step_sentences(instructions_raw, instructions_lexed):
    conjunctions =[',and','then','and then']
    tokens_raw = []
    tokens_lexed = []
    assert len(instructions_lexed) == len(instructions_raw), "Error in combining the instructions in dataset.py"
    
    for i in range(len(instructions_lexed)):
        this_lexed = [ w for w in instructions_lexed[i].split() if w not in ['<BOS>', '<EOS>']]
        this_raw = instructions_raw[i].split()
        assert len(this_lexed) == len(this_raw), "There is a length mismatch between raw and lexed instructions. please disable group concepts"
        if i+1 != len(instructions_lexed):
            conj = random.choice(conjunctions)
            this_lexed.append(conj)
            this_raw.append(conj)
        tokens_raw.extend(this_raw)
        tokens_lexed.extend(this_lexed)
    return tokens_raw, tokens_lexed

