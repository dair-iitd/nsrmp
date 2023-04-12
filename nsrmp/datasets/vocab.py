#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : vocab.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Modifier: Namasivayam K
# Email : Namasivayam.k@cse.iitd.ac.in 
# Modified date   : 09/06/2022

#Acknowledgement: Code adopted from NSCL-Pytorch release by Jiayuan Mao

import helpers.io as io
from typing import Callable
# from collections.abc import Callable
from helpers.logging import get_logger

logger = get_logger(__file__)

class Vocab(object):
	def __init__(self, word2idx=None):
		self.word2idx = word2idx if word2idx is not None else dict()
		self._idx2word = None

	@classmethod
	def from_json(cls, json_file):
		return  cls(io.load_json(json_file))
		
	def from_dataset(get_sent_tokenized, dataset_length, extra_words,vocab_cls, save_vocab = False):
		return gen_vocab(get_sent_tokenized,dataset_length,extra_words, vocab_cls, save_vocab)

	def dump_json(self, json_file):
		io.dump_json(json_file, self.word2idx)

	def check_json_consistency(self, json_file):
		rhs = io.load_json(json_file)
		for k, v in self.word2idx.items():
			if not (k in rhs and rhs[k] == v): return False
		return True

	def words(self):
		return self.word2idx.keys()

	@property
	def idx2word(self):
		if (self._idx2word is None) or (len(self.word2idx) != len(self._idx2word)):
			self._idx2word = {v: k for k, v in self.word2idx.items()}
		return self._idx2word

	def __len__(self):
			return len(self.word2idx)

	def __iter__(self):
			return iter(self.word2idx.keys())

	def add(self, word):
			self.add_word(word)

	def add_word(self, word):
		self.word2idx[word] = len(self.word2idx)
	
	def map(self, word):
		return self.word2idx.get(word, self.word2idx.get('<UNK>', -1))

	def map_sequence(self, sequence):
		if isinstance(sequence, str):
			sequence = sequence.split()
		return [self.map(w) for w in sequence]

def gen_vocab(get_sent_tokenized:Callable[[int],list],num_sent, extra_words, vocab_cls, save_vocab = False):
	if vocab_cls is None:
		vocab_cls = Vocab
	all_words = set()
	for idx in range(num_sent):
		sent = get_sent_tokenized(idx)
		for w in sent:
			all_words.add(w)
	vocab = vocab_cls()
	vocab.add('<PAD>')
	for w in ['<UNK>','<EOS>','<BOS>']:
		vocab.add(w)

	for w in sorted(all_words):
		vocab.add(w)

	if extra_words is not None:
		for w in extra_words:
			vocab.add(w)
	if save_vocab == True:
		logger.critical("Saving vocab to vocab_new.json")
		vocab.dump_json('vocab_new.json')		

	return vocab
