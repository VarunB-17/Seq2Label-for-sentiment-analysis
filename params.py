"""
Hyper-parameters for binary classification of text
where all values are some power of 2
"""
EPOCHS = 10  # amount of epochs
LR = 0.001  # learning rate
MAX_TOKENS = 8174  # token limit of a batch
CLS = 2  # classes
VOCAB_SIZE = 99430  # vocabulary size
EMBED_DIM = 512  # embedding dimension
POOL = 'avg'  # type of pooling
ATTENTION = 'mha'  # type of attention
HEADS = 4  # number of attention heads
DROPOUT = 0.0  # dropout rate
HIDDEN = 2  # amount of hidden layers for ff
ENCDEPTH = 4  # amount of encoder layers
DECDEPTH = 4  # amount of decoder layer
MASK = True
