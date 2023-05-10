from torch import nn
from attention import MultiHeadedAttention, PositionalEmbedding
from params import *


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, heads=4, dropout=0.0, hidden=2):
        super().__init__()
        self.hidden = hidden
        self.attention = MultiHeadedAttention(input_dim=input_dim, heads=heads, mask=False)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, self.hidden * input_dim),
            nn.ReLU(),
            nn.Linear(self.hidden * input_dim, input_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # multi-headed attention
        mha = self.attention(x)
        # layer normalization by attention and residual connection
        x = self.norm1(mha + x)
        # apply dropout
        x = self.dropout1(x)
        # pass through feedforward layer
        ff = self.feedforward(x)
        # apply layer normalization
        x = self.norm2(ff + x)
        # apply dropout
        x = self.dropout2(x)
        # return encoded embedding
        return x


class DecoderLayer(nn.Module):
    # TODO yet to be implemented!

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):

    def __init__(self, embed_dim, heads, dropout, hidden, depth):
        super().__init__()
        self.depth = depth
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.position = PositionalEmbedding(embed_dim=embed_dim)

        # apply encoder stacks
        enc_block = []
        for i in range(depth):
            enc_block.append(
                EncoderLayer(input_dim=embed_dim,
                             dropout=dropout,
                             hidden=hidden,
                             heads=heads
                             )
            )
        self.encoder_block = nn.Sequential(*enc_block)

    def forward(self, x):
        # convert sequence to embeddings + pe
        x = self.embedding(x)
        x = self.position(x)
        # apply encoder stacks
        x = self.encoder_block(x)

        return x


class Decoder(nn.Module):
    # TODO yet to be implemented!

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
