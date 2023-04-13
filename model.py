import torch
import torch.nn as nn
from params import CLS,EMBED_DIM


class SentimentCLF(torch.nn.Module):
    """
    Text classification for sentiment
    analysis using a embedding layer
    to goes into a pooling layer
    (avg,max) followed by conversion
    to a vector of class probabilities
    """

    def __init__(self, nr_embed, embed_dim=EMBED_DIM, output_dim=CLS):
        super(SentimentCLF, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.nr_embed = nr_embed

        self.embedding = nn.Embedding(nr_embed, embed_dim)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.out = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pool(x)
        x = self.out(x)

        return x
