import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import CLS

EMBED_DIM, CLS = 300, CLS


class SentimentCLF(torch.nn.Module):
    def __init__(self, nr_embed, embed_dim=EMBED_DIM, output_dim=CLS):
        super(SentimentCLF, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.nr_embed = nr_embed
        self.embedding = nn.Embedding(nr_embed, embed_dim)
        self.linear1 = nn.Linear(embed_dim, 128)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.pool(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = self.linear2(x)
        x = self.out(x.squeeze())

        return x
