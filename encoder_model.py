from params import *
from torch import nn
from transformer_sub import Encoder
import torch


class EncoderModel(nn.Module):
    """""
    Encoder classification model
    for sentiment analysis.
    """""
    def __init__(self,
                 nr_embed=VOCAB_SIZE,
                 embed_dim=EMBED_DIM,
                 output_dim=CLS,
                 pool_type=POOL,
                 attention=ATTENTION,
                 heads=HEADS,
                 dropout=DROPOUT,
                 hidden=HIDDEN,
                 depth=ENCDEPTH):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.nr_embed = nr_embed
        self.attention = attention
        self.heads = heads
        self.pool_type = pool_type
        self.dropout = dropout
        self.depth = depth
        self.encoder = Encoder(embed_dim=embed_dim,
                               heads=heads,
                               dropout=dropout,
                               hidden=hidden,
                               depth=depth)
        self.out = nn.Linear(in_features=embed_dim, out_features=output_dim)

    def forward(self, x):
        # Encoder block
        x = self.encoder(x)
        # Pooling
        x = torch.mean(x, dim=1)
        # class probabilities
        x = self.out(x)
        # output
        return x
