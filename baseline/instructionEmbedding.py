########################
### Language Encoder ###
########################

# PyTorch related imports
import torch
import torch.nn as nn

# Jacinle related imports
import jactorch.nn as jacnn
from jactorch.nn.embedding import LearnedPositionalEmbedding


# Module for encoding input instruction
class InstructionEmbedding(nn.Module):

    # Arguments:
    #   vocab: (Vocab)           : Word vocabulary (vocab.json)
    #   gru_hidden_dim: (int)    : Hidden layer dimension of GRU (512)
    #   gru_nlayers: (int)       : Number of GRU layers (2)
    #   word_emb_dim: (int)      : Word embedding dimension (256)
    #   positional_emb_dim: (int): Word positional embedding dimension (50)
    def __init__(self, vocab, gru_hidden_dim, gru_nlayers, word_emb_dim, positional_emb_dim):
        super().__init__()
        self.vocab = vocab
        self.hidden_dim = gru_hidden_dim
        self.nlayers = gru_nlayers
        self.word_embedding_dim = word_emb_dim
        self.positional_embedding_dim = positional_emb_dim
        self.total_embedding_dim = self.word_embedding_dim
        self.word_embedding = nn.Embedding(len(vocab), self.word_embedding_dim, padding_idx=0)
        if self.positional_embedding_dim is not None:
            self.positional_embedding = LearnedPositionalEmbedding(128, self.positional_embedding_dim)
            self.total_embedding_dim += self.positional_embedding_dim
        self.gru = jacnn.GRULayer(self.total_embedding_dim, self.hidden_dim, self.nlayers, batch_first=True, bidirectional=True)
    
    # Arguments:
    #   sent: (tensor(B X L))    : Batch of input instructions (after word2idx)
    #   sent_length: (tensor(B,)): Batch of length of given instruction
    # Return Value:
    #   last_state: (tensor(B X 1024)): Batch of dense encoding for each instruction
    # Here, B = batch size and L = maximum length of instruction in a given batch
    def forward(self, sent, sent_length):
        f = self.word_embedding(sent)
        if self.positional_embedding is not None:
            f = torch.cat([f, self.positional_embedding(sent)], dim=-1)
        _, last_state = self.gru(f, sent_length)
        return last_state
