"""
Loads the IMDB dataset as a pandas dataframe
in order to perform basic visualization.
"""
import pandas as pd
import numpy as np
from data_rnn import load_imdb
import torch
from torch.utils.data import Dataset, DataLoader


def data2df(val) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    df_train['x_train'] = df_train['x_train'].apply(lambda x: torch.tensor(x, dtype=torch.long, device=cuda0))
    df_train['y_train'] = df_train['y_train'].apply(lambda x: torch.tensor(x, dtype=torch.long, device=cuda0))

    return df_train, df_test


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)
    return device


def padding(df):

    df
    return None


class DataSentiment(Dataset):
    def __init__(self, val=False):
        self.val = val
        self.df = data2df(val)
        self.x_train = padding(self.df[0]['x_train'])
        self.y_train = padding(self.df[0]['y_train'])

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

cuda0 = get_device()
data = DataSentiment(val=False)
print(data.x_train[0])
print(type(data.x_train))
print(type(data.x_train[0]))

np.array_split(df, 3)