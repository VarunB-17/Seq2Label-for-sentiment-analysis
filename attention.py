import torch
from params import EMBED_DIM
import torch.nn.functional as F


class PositionalEmbedding:
    """
    Addition of a vector to
    a word embedding based on
    its relative positioning
    using the sine and cosine
    functions and the dimension
    of an embedding vector.
    """

    def __init__(self, sequence, embed_dim=EMBED_DIM):
        #  define params
        self.sequence = sequence
        self.embed_dim = embed_dim

    def embedding(self):
        #  get amount of words in sequence
        seq_len = len(self.sequence)
        # tensor of length embedding dimension
        pos_even = torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
        #  compute denominator
        denominator = torch.pow(10000, pos_even/self.embed_dim)
        #  define empty positional embedding
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        #  compute embedding values for odd and even positional
        even = torch.sin(pos / denominator)
        odd = torch.cos(pos / denominator)
        #  interleave odd and even matrices among second dimension
        pos_embed = torch.stack([even, odd], dim=2)
        pos_embed = torch.flatten(pos_embed, start_dim=1, end_dim=2)
        #  add positional embedding to word embedding
        embedding = self.sequence + pos_embed
        return embedding


class SimpleSelfAttention:
    """
    Computing the attention scores
    for each sequence in a batch of
    sequences. Inspired by simple
    self-attention as proposed
    by dr. Bloem
    """
    def __init__(self, x):
        self.x = x

    def attention(self):
        weights = torch.bmm(self.x, self.x.transpose(1, 2))
        weights = F.softmax(weights, dim=2)
        x_attention = torch.bmm(self.x.transpose(1, 2), weights)
        return x_attention


class MultiHeadedAttention:
    def __init__(self, x):
        self.x = x
