import pandas as pd
from data_rnn import load_imdb
import torch


def data2df(final) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the raw data into a pandas dataframe,
    and converts the list with integers into tensors.
    """
    # load data
    (x_train, y_train), (x_test, y_test), (i2w, w2i), _ = load_imdb(final=final)

    # Convert word ids to actual string representation of words (for representation)
    words_train = [[i2w[word] for word in sent] for sent in x_train]
    words_test = [[i2w[word] for word in sent] for sent in x_test]

    # Dataframes of training and test/validation data
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

    # convert training and test/validation data from lists to tensors
    df_train['x_train'] = df_train['x_train'].apply(lambda x: torch.tensor(x, dtype=torch.long))
    df_train['y_train'] = df_train['y_train'].apply(lambda x: torch.tensor(x, dtype=torch.long))
    df_test['x_test'] = df_test['x_test'].apply(lambda x: torch.tensor(x, dtype=torch.long))
    df_test['y_test'] = df_test['y_test'].apply(lambda x: torch.tensor(x, dtype=torch.long))

    # return training data and validation/test data
    return df_train, df_test


def padding(batch):
    """
    pads batches according to the longest sequence
    in a batch and truncates each batch until max_tokens
    is reached.
    """

    padded_batch = list()
    max_tensor = max([len(x[0]) for x in batch])  # get longest tensor
    pad = [max_tensor - len(x[0]) for x in batch]  # how much padding is required for each sequence

    # only apply padding is the sum of the pad list is not 0
    if sum(pad) != 0:
        for i in range(len(pad)):
            if pad[i] > 0:
                padded_batch.append((torch.cat((batch[i][0], torch.zeros(pad[i]))), batch[i][1]))  # pad
            else:
                padded_batch.append(batch[i])  # append sequence if no padding is required
    else:
        padded_batch = batch

    # compute amount of tokens in the padded batch
    batch_length = sum([len(x[0]) for x in padded_batch])

    # return batch and length of the batch
    return padded_batch, batch_length


def formatted(batch):
    """
    Format each batch such that each batch consist of 2 tensors.
    The first tensors is a tensor consisting of multiple
    tensors that share the same dimension. On the other
    hand, the second tensor is 1 dimensional and holds all
    class labels.
    """
    sequences = torch.stack([x[0] for x in batch], dim=0)
    labels = torch.stack([x[1] for x in batch])
    batch = [sequences, labels]
    print(batch)
    return batch


def get_device():
    """
    Checks if gpu is available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)
    return device
