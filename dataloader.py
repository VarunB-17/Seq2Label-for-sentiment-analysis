"""
Loads the IMDB dataset as a pandas dataframe
in order to perform basic visualization.
"""
import pandas as pd
from data_rnn import load_imdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

CLS, BATCHES, MAX_TOKENS = 2, 512, 64*2*2*2*2*2*2*2


def data2df(val) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the raw data into a pandas dataframe,
    and converts the list with integers into tensors.
    """
    # load data
    (x_train, y_train), (x_test, y_test), (i2w, w2i), _ = load_imdb(final=val)

    # Convert word ids to actual string representation of words
    words_train = [[i2w[word] for word in sent] for sent in x_train]
    words_test = [[i2w[word] for word in sent] for sent in x_test]

    # Dataframe
    df_train = pd.DataFrame({'sent': words_train,
                             'x_train': x_train,
                             'y_train': y_train})

    df_test = pd.DataFrame({'sent': words_test,
                            'x_test': x_test,
                            'y_test': y_test})

    # sorting data in ascending order
    df_train['len'] = df_train['x_train'].apply(lambda x: len(x))
    df_train = df_train.sort_values(by=['len'])

    df_test['len'] = df_test['x_test'].apply(lambda x: len(x))
    df_test = df_test.sort_values(by=['len'])

    # convert list to tensor
    df_train['x_train'] = df_train['x_train'].apply(lambda x: torch.tensor(x, dtype=torch.long))
    df_train['y_train'] = df_train['y_train'].apply(lambda x: torch.tensor(x, dtype=torch.long))

    df_test['x_test'] = df_test['x_test'].apply(lambda x: torch.tensor(x, dtype=torch.long))
    df_test['y_test'] = df_test['y_test'].apply(lambda x: torch.tensor(x, dtype=torch.long))

    return df_train, df_test


def padding(batches):
    """
    pads batches according
    to the longest sequence
    in a batch and truncates
    each batch until max_tokens
    is reached.
    """

    padded_batches = []
    for batch in batches:
        sequences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        padded_seq = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        padded_batches.append((padded_seq, torch.tensor(labels)))

    return padded_batches


def get_device():
    """
    Run on cpu
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)
    return device


class DatasetSentiment(Dataset):
    """
    Custom Dataset class for
    the IMDB reviews dataset.
    """

    def __init__(self, val=False):
        self.val = val
        self.df = data2df(val)
        self.x_train = self.df[0]['x_train']
        self.y_train = self.df[0]['y_train']
        self.x_test = self.df[1]['x_test']
        self.y_test = self.df[1]['y_test']

        vocab_dimension = torch.tensor([0])
        for tens in self.x_train.values:
            vocab_dimension = torch.cat((vocab_dimension, tens), dim=0)
            vocab_dimension = torch.tensor([vocab_dimension.max().item()])

        self.vocab_size = vocab_dimension

    def __len__(self):
        """
        returns the amount of instances
        present in the training data.
        """
        return len(self.x_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


class DynamicBatchLoader(DataLoader):
    """
    Creates dynamic batches from
    the DatasetSentiment class object.
    """

    def __init__(self, dataset, max_tokens, batch_size):
        self.max_tokens = max_tokens
        super().__init__(dataset, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=False)

    def collate_fn(self, batch):
        batches = []
        current_batch = []
        current_tokens = 0

        for sequence, label in batch:
            seq_len = len(sequence)

            if seq_len + current_tokens > self.max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append((sequence, label))
            current_tokens += seq_len

        if current_batch:
            batches.append(current_batch)

        padded_batches = padding(batches)
        return padded_batches
