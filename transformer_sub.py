from torch import nn
from attention import MultiHeadedAttention

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # multi-headed attention
        mha = self.attention(x)
        # layer normalization by attention and residual connection
        x = self.norm1(mha + x)
        # apply dropout
        x = self.dropout(x)
        # pass through feedforward layer
        ff = self.feedforward(x)
        # apply layer normalization
        x = self.norm2(ff + x)
        # apply dropout
        x = self.dropout(x)
        # return encoded embedding
        return x


class DecoderLayer(nn.Module):
    # TODO yet to be implemented!

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    # TODO yet to be implemented!

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x




class Decoder(nn.Module):
    # TODO yet to be implemented!

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
