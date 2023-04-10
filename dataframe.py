"""
Loads the IMDB dataset as a pandas dataframe
in order to perform basic visualization.
"""
import pandas as pd
import numpy as np
from data_rnn import load_imdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad

CLS, BUCKETS, BATCHES, MAX_TOKENS = 2, 256, 64, 16384


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

    return df_train, df_test


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


def padding(df):
    """

    """
    buckets = np.array_split(df, BUCKETS)
    sequence = []

    for bucket in buckets:
        print(len(bucket))

    for bucket in buckets:
        max_tokens = (bucket.tail(1).iloc[0]).size()[0]
        for ts in bucket:
            padded_tensor = pad(ts, (0, max_tokens))
            sequence.append(padded_tensor)

    return sequence


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

    def __len__(self):
        """
        returns the amount of instances
        present in the training data.
        """
        return len(self.x_train)

    def __getitem__(self, index):
        """
        return
        """

        return self.x_train[index], self.label[index]

class DynamicBatch(DataLoader):
    """
    Creates dynamic batches from
    the DatasetSentiment class object.
    """

    def __init__(self,dataset,max_tokens,buckets,batches):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.buckets = buckets
        self.batches = batches