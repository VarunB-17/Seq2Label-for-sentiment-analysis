"""
Loads the IMDB dataset as a pandas dataframe
in order to perform basic visualization.
"""
import pandas as pd
from data_rnn import load_imdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad

CLS, BATCHES, MAX_TOKENS, BUCKETS = 2, 64, 16384, 256


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

def truncate(batches):
    """
    truncates a list of tensor
    to the length of the required
    amount of tokens allowed for
    each batch in the set of batches.
    """


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
        max_tensor = sequences[-1].size()[0]
        padded_seq = [pad(seq, (0, max_tensor), 'constant', 0) for seq in sequences]
        padded_batches.append(padded_seq)

    truncated_batches = truncate(padded_batches)

    return truncated_batches

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

        return self.x_train[index], self.y_train[index]


class DynamicBatchLoader(DataLoader):
    """
    Creates dynamic batches from
    the DatasetSentiment class object.
    """

    def __init__(self, dataset, max_tokens, batches, bucket_size):
        super().__init__(dataset, batch_size=batches, collate_fn=self.collate_fn)
        self.max_tokens = max_tokens
        self.bucket_size = bucket_size

    def collate_fn(self, batch):
        """
        creates batches using dynamic batching
        with buckets where each batch is filled
        with the instances of a bucket until
        the max_tokens property is reached.
        source: https://rashmi-margani.medium.com/how-to-speed-up-the-training-
        of-the-sequence-model-using-bucketing-techniques-9e302b0fd976
        """

        # create buckets from the sorted data
        buckets = []
        for i in range(0, len(batch), self.bucket_size):
            bucket = batch[i:i + self.bucket_size]
            buckets.append(bucket)

        # dynamic length batches using the buckets and max_tokens
        batches = []
        for bucket in buckets:
            batch = []
            token_count = 0
            for sequence, label in bucket:
                len_seq = sequence.size()[0]
                if len_seq + token_count > self.max_tokens:
                    batches.append(batch)
                    batch = []  # reset batch list
                    token_count = 0  # reset token count
                batch.append((sequence, label))
                token_count += len_seq
            # handles cases where a bucket is empty
            if len(batch) > 0:
                batches.append(batch)

        self.padded_sequences = padding(batches)

        return self.padded_sequences
