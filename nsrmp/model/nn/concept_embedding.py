#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   embeddings.py
#Time    :   2022/06/22 19:51:44
#Contributors : Himanshu G Singh, Namasivayam K, 
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

#Acknowledgement: Part of this code is adopted from NSCL-Pytorch Release.

import torch
import torch.nn as nn
import torch.nn.functional as F
from cached_property import cached_property
from datasets.definition import gdef

class ActionBlock(nn.Module):
    pass

class AttributeBlock(nn.Module):
    def __init__(self, object_feature_dim, attr_emb_dim, hidden_dim = 64):
        super().__init__( )
        self.feature_dim = object_feature_dim
        self.output_dim = attr_emb_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2*object_feature_dim
        self.map = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim,self.output_dim)
        )
    def forward(self,x):
        return self.map(x)


class ConceptBlock(nn.Module):
    """
    Concept as an embedding in the corresponding attribute space.
    """
    def __init__(self, embedding_dim, nr_attributes, attribute_agnostic=False):
        """

        Args:
            embedding_dim (int): dimension of the embedding.
            nr_attributes (int): number of known attributes.
            attribute_agnostic (bool): if the embedding in different embedding spaces are shared or not.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.nr_attributes = nr_attributes
        self.attribute_agnostic = attribute_agnostic

        if self.attribute_agnostic:
            self.embedding = nn.Parameter(torch.randn(embedding_dim))
        else:
            self.embedding = nn.Parameter(torch.randn(nr_attributes, embedding_dim))
        self.belong = nn.Parameter(torch.randn(nr_attributes) * 0.1)

        self.known_belong = False

    def set_belong(self, belong_id):
        """
        Set the attribute that this concept belongs to.

        Args:
            belong_id (int): the id of the attribute.

        """
        self.belong.data.fill_(-100)
        self.belong.data[belong_id] = 100
        self.belong.requires_grad = False
        self.known_belong = True

    @property
    def normalized_embedding(self):
        """L2-normalized embedding in all spaces."""
        embedding = self.embedding / self.embedding.norm(2, dim=-1, keepdim=True)
        if self.attribute_agnostic:
            embedding = embedding.expand(self.nr_attributes,-1)
        return embedding

    @property
    def log_normalized_belong(self):
        """Log-softmax-normalized belong vector."""
        return F.log_softmax(self.belong, dim=-1)

    @property
    def normalized_belong(self):
        """Softmax-normalized belong vector."""
        return F.softmax(self.belong, dim=-1)


class ConceptEmbeddings(nn.Module):
    def __init__(self,attribute_agnostic = False):
        super().__init__()
        self.attribute_agnostic = attribute_agnostic
        self.all_attributes = []
        self.all_concepts = []
        self.attribute_operators = nn.Module()
        self.concept_embeddings = nn.Module()
    
    @cached_property
    def nr_attributes(self):
        return len(self.all_attributes)

    @cached_property
    def nr_concepts(self):
        return len(self.all_concepts)

    @cached_property
    def attribute2id(self):
        return {a:i for i,a in enumerate(self.all_attributes)}
    
    @cached_property
    def id2attribute(self):
        return {i:a for i,a in enumerate(self.all_attributes)}

    def init_attribute(self, identifier, input_dim, output_dim):
        assert self.nr_concepts == 0, 'Can not register attributes after having registered any concepts.'
        self.attribute_operators.add_module('attribute_' + identifier, AttributeBlock(object_feature_dim = input_dim, attr_emb_dim = output_dim))
        self.all_attributes.append(identifier)

    def init_concept(self, identifier, emb_dim, known_belong=None):
        block = ConceptBlock(emb_dim, self.nr_attributes, attribute_agnostic=self.attribute_agnostic)
        self.concept_embeddings.add_module('concept_' + identifier, block)
        if known_belong is not None:
            block.set_belong(self.attribute2id[known_belong])
        self.all_concepts.append(identifier)
 
    def get_belong(self, identifier):
        belong_score = getattr(getattr(self.concept_embeddings, 'concept_'+ identifier), 'belong')
        return self.all_attributes[belong_score.argmax(-1).item()]
    
    def get_all_belongs(self):
        belongs = dict()
        for k, v in self.concept_embeddings.named_children():
            belongs[k] = self.all_attributes[v.belong.argmax(-1).item()]
        return belongs
    
    def get_attribute_operator(self, identifier):
        x = getattr(self.attribute_operators, 'attribute_' + identifier)
        return x.map
    
    @cached_property
    def get_all_attribute_operators(self):
        return [self.get_attribute_operator(a) for a in self.all_attributes]
    
    def get_concept(self, identifier):
        return getattr(self.concept_embeddings, 'concept_'+identifier)
    
    def init_from_dict(self,used_concepts_dict, configs):
        pass
    
    def init_from_gdef(self, type, object_feature_dim, emb_dim):
        attr_dict = getattr(gdef, type)
        for attr in attr_dict:
            self.init_attribute(attr,object_feature_dim,emb_dim)
        for attr in attr_dict:
            for concept in attr_dict[attr]:
                self.init_concept(concept,emb_dim,known_belong=attr)
        





