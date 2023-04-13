# imports
from dataloader import *
from model import SentimentCLF
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from params import EPOCHS, PROGRESS, LR, MAX_TOKENS


def data(load=True):
    """
    Call this function if you want to regenerate
    the same 'batches.pt' file locally.
    In this way you don't need to
    rerun this function everytime
    the script is ran.
    """
    # Define training and eval/test data
    if load:
        data_train, data_eval = data2df(final=False)
        data_train, data_eval = DatasetSentiment(data_train), DatasetSentiment(data_eval)

        # Define batches
        batches_loader = DynamicBatchLoader(dataset=data_train, max_tokens=MAX_TOKENS, debug=True).dynamic()
        return torch.save(batches_loader, 'batches.pt')


data(load=False)  # set to True if you want to generate the file locally

batches = torch.load('batches.pt')

for batch in batches:
    print(batch[0].size())
# # Define validation,model,loss, and optimizer
# validation = (data_training.x_test, data_training.y_test)
# model = SentimentCLF(nr_embed=data_training.vocab_size)
# loss_func = CrossEntropyLoss()
# optimizer = SGD(model.parameters(), lr=LR)
#
#
# def train(epochs):
#     """
#     Trains a pytorch model that
#     stops training if after x
#     epochs there ha
#     """
#     for epoch in range(epochs):
#         current_loss = 0.0
#         for batch in batches:
#             optimizer.zero_grad()
#
#             outputs = model(inputs)
#             loss = loss_func(outputs, label)
#             loss.backward()
#             optimizer.step()
#
#             if (i + 1) % PROGRESS == 0:
#                 print(
#                     f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{i + 1}/{len(data_training.x_train)}], Loss: {loss.item():.4f}")
#
#             current_loss += loss.item()
#
#         epoch_loss = current_loss / len(data_training.x_train)
#         print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
#
#
# def eval():
#     """
#     Evaluate the trained model
#     with an evaluation dataset
#     to check the performance
#
#     """
#     pass
