import torch
import torch.nn as nn
from params import CLS, EMBED_DIM
from attention import SimpleSelfAttention, MultiHeadedAttention, PositionalEmbedding


class SentimentCLF(torch.nn.Module):
    """
    Text classification for sentiment analysis using a embedding layer
    to goes into a pooling layer (avg,max) followed by conversion
    to a vector of class probabilities
    """

    def __init__(self, nr_embed, embed_dim=EMBED_DIM, output_dim=CLS, pool_type='avg', attention='simple',pe=True):
        super(SentimentCLF, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.nr_embed = nr_embed
        self.attention = attention
        self.pe = pe
        self.embedding = nn.Embedding(nr_embed, 4)
        self.pool = pool_type
        self.out = nn.Linear(embed_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.embedding(x)

        if self.pe:
            x = PositionalEmbedding(x).embedding()

        if self.attention == 'simple':
            x = SimpleSelfAttention(x).attention()
        elif self.attention == 'mha':
            x = MultiHeadedAttention(x)
        else:
            raise ValueError("attention type has not been specified")

        if self.pool == 'avg':
            x = torch.mean(x, dim=1)
        elif self.pool == 'max':
            x = torch.max(x, dim=1)[0]
        else:
            raise ValueError('pooling type has not been specified')

        x = self.out(x)

        return x
