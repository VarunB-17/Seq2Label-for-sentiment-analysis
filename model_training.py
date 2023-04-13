# imports
from dataloader import *
from model import SentimentCLF
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from params import EPOCHS, PROGRESS, LR, MAX_TOKENS

# Define training data
data_training = DatasetSentiment(final=False)  # evaluate training data on evaluation set (75/25) split

# Define batches
batches = DynamicBatchLoader(dataset=data_training, max_tokens=MAX_TOKENS).dynamic()

# Define validation,model,loss, and optimer
validation = (data_training.x_test, data_training.y_test)
model = SentimentCLF(nr_embed=data_training.vocab_size)
loss_func = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)


def train(epochs):
    """
    Trains a pytorch model that
    stops training if after x
    epochs there ha
    """
    for epoch in range(epochs):
        current_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()

            if (i + 1) % PROGRESS == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{i + 1}/{len(data_training.x_train)}], Loss: {loss.item():.4f}")

            current_loss += loss.item()

        epoch_loss = current_loss / len(data_training.x_train)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")


def eval():
    """
    Evaluate the trained model
    with an evaluation dataset
    to check the performance

    """
    pass
