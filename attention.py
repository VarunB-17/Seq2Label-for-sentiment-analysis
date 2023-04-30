import torch
from torch import nn
from params import EMBED_DIM
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """
    Addition of a vector to
    a word embedding based on
    its relative positioning
    using the sine and cosine
    functions and the dimension
    of an embedding vector.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # tensor of length embedding dimension
        pos_even = torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
        #  compute denominator
        denominator = torch.pow(10000, pos_even / self.embed_dim)
        #  define empty positional embedding
        pos = torch.arange(x.size(1), dtype=torch.float32).unsqueeze(1)
        #  compute embedding values for odd and even positional
        even = torch.sin(pos / denominator)
        odd = torch.cos(pos / denominator)

        #  interleave odd and even matrices among second dimension
        pos_embed = torch.stack([even, odd], dim=2)
        pos_embed = torch.flatten(pos_embed, start_dim=1, end_dim=2)
        # broadcast along shape of batch size in order to match dimensions of input seq
        pos_embed = pos_embed.expand(x.size(0), -1, -1)
        #  add positional embedding to word embedding
        embedding = x + pos_embed
        return embedding


def simple_attention(x):
    """
    Computing the attention scores
    for each sequence in a batch of
    sequences. Inspired by simple
    self-attention as proposed
    by dr. Bloem
    """
    # weighted sum of the dot-products of each embedding with every other embedding
    weights = torch.bmm(x, x.transpose(1, 2))
    # apply softmax over each embedding
    weights = F.softmax(weights, dim=2)
    # update each embedding by attention weights
    x_attention = torch.bmm(weights, x)
    return x_attention


def att(q, k, v, mask=False):
    d_k = q.size(3)
    # query T key
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # apply masking
    if mask:
        mask = torch.full(scaled.size(), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        scaled += mask
    # apply softmax
    attention = F.softmax(scaled, dim=3)
    # multiply qTk by value vector
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dim, heads=4, mask=False):
        super().__init__()
        assert input_dim % heads == 0
        self.input_dim = input_dim
        self.heads = heads
        self.head_dim = input_dim // heads
        self.mask = mask

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        hd = self.head_dim
        assert k == self.input_dim
        # define linear transformation to project sequence to query,key,value vectors
        concat_qkv = nn.Linear(k, 3 * k, bias=False)
        # apply transformation
        concat_qkv = concat_qkv(x)  # torch.Size([190, 43, 1536])
        # reshape last dimension to number of heads and head dimension * 3
        concat_qkv = concat_qkv.reshape(b, t, h, 3 * hd)  # torch.Size([190, 43, 4, 384])
        # swap second and third dimension
        concat_qkv = concat_qkv.permute(0, 2, 1, 3)  # torch.Size([190, 4, 43, 384])
        # break tensor by last dim to obtain the separate query,key,value vector
        query, key, value = concat_qkv.chunk(3, dim=-1)
        # apply attention
        values, attention = att(query, key, value, mask=self.mask)
        # concat all attention head
        values = values.reshape(b, t, h * hd)
        # output vector
        out_linear = nn.Linear(k, k, bias=False)
        out = out_linear(values)
        return out
