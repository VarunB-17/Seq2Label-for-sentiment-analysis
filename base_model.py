import torch
import torch.nn as nn
from params import CLS, EMBED_DIM, VOCAB_SIZE, ATTENTION, HEADS, POOL, LR, MASK,PE,EPOCHS
from attention import MultiHeadedAttention, PositionalEmbedding, simple_attention
from batch_functions import get_device
import torch.nn.functional as F
device = get_device()


class SentimentCLF(nn.Module):
    """
    Text classification for sentiment analysis using a embedding layer
    to goes into a pooling layer (avg,max) followed by conversion
    to a vector of class probabilities
    """

    def __init__(self,
                 nr_embed=VOCAB_SIZE,
                 embed_dim=EMBED_DIM,
                 output_dim=CLS,
                 pool_type='avg',
                 attention='none',
                 pe=False,
                 mask=False,
                 heads=4):
        super(SentimentCLF, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.nr_embed = nr_embed
        self.attention = attention
        self.position = pe
        if self.position:
            self.pe = PositionalEmbedding(embed_dim=self.embed_dim)
        self.mask = mask
        self.embedding = nn.Embedding(nr_embed, embed_dim)
        self.pool = pool_type
        self.heads = heads
        if ATTENTION == 'mha':
            self.mha = MultiHeadedAttention(heads=self.heads, mask=self.mask, input_dim=self.embed_dim)
        self.out = nn.Linear(embed_dim, output_dim, bias=False)

    def forward(self, x):
        x = x.to(device)
        # embedding layer
        x = self.embedding(x)

        # apply positional embedding
        if self.position:
            x = self.pe(x)
        # specify type of self attention
        if self.attention == 'simple':
            x = simple_attention(x)
        elif self.attention == 'mha':
            x = self.mha(x)
        elif self.attention == 'none':
            pass
        else:
            raise ValueError("attention type has not been specified")

        # max or average pooling operation
        if self.pool == 'avg':
            x = torch.mean(x, dim=1)
        elif self.pool == 'max':
            x = torch.max(x, dim=1)[0]
        else:
            raise ValueError('pooling type has not been specified')

        # convert pooling output to 2 class probabilities
        x = self.out(x)

        x = F.softmax(x, dim=1)

        return x
