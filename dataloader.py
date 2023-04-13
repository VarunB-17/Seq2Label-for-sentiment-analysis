"""
Loads the IMDB dataset as a pandas dataframe
in order to perform basic visualization.
"""
import torch
from torch.utils.data import Dataset
from params import MAX_TOKENS
from batch_functions import formatted, padding, data2df, get_device, load_imdb

class DatasetSentiment(Dataset):
    """
    Custom Dataset class for
    the IMDB reviews dataset.
    """

    def __init__(self, final=False):
        self.val = final
        self.df = data2df(final)
        self.x_train = self.df[0]['x_train']
        self.y_train = self.df[0]['y_train']
        self.x_test = self.df[1]['x_test']
        self.y_test = self.df[1]['y_test']

        # returns the size of the vocabulary
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
        """
        returns a sequence and
        label at some index -> [index]
        """
        return self.x_train[index], self.y_train[index]


class DynamicBatchLoader:
    """
    Creates dynamic sizes batches where
    each batch is filled with sequences
    until some token limit is exceeded.
    When the token limit is reached
    it will move to a new batch
    """

    def __init__(self, dataset, max_tokens=MAX_TOKENS):
        self.max_tokens = max_tokens # define maximum tokens allowed for each batch
        self.dataset = dataset # original dataset

    def dynamic(self):
        """
        returns batches according to the
        method described in the class.
        there are two cases when a batch
        gets added, where in the first case
        the addition of a sequence to a list
        of padded sequences exceeds the token
        limit, and a second case where if
        a sequence without padding is under
        the token limit but after padding
        exceeds the limit.
        This will allow for batches that will
        never exceed the token limit and this
        includes padded sequences.
        """

        # Initiate empty batches
        batches = list()
        current_batch = list()
        batch_length = int(0)

        # loop through the dataset
        for i in range(len(self.dataset)):
            (sequence, label) = self.dataset[i]

            # get the length of the sequence
            sequence_length = sequence.size()[0]

            # case where without padding limit is exceeded
            if batch_length + sequence_length > self.max_tokens:
                x = [x[0] for x in current_batch]
                print(f'{x[0].size()} {len(x)} - without padding')
                batches.append(formatted(current_batch))
                current_batch = list()

            # temporary addition of a sequence only if including padded it still hold the max_token property
            temp_seq = current_batch.copy()
            temp_seq.append((sequence, label))
            temp_batch, batch_length = padding(temp_seq)

            # case where with padding limit is exceeded
            if batch_length > self.max_tokens:
                x = [x[0] for x in current_batch]
                print(f'{x[0].size()} {len(x)} with padding')
                batches.append(formatted(current_batch))
                current_batch = list()
                batch_length = int(0)
            else:
                current_batch = temp_batch

        # add the last batch after iteration has been completed
        if current_batch:
            x = [x[0] for x in current_batch]
            print(f'{x[0].size()} {len(x)} last batch')
            batches.append(formatted(current_batch))

        return batches
