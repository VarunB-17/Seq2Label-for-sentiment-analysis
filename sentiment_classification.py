from dataloader import *
from model import SentimentCLF
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# Basic setup
data_training = DatasetSentiment(val=False)  # evaluate training data on evaluation set (75/25) split
batches = DynamicBatchLoader(dataset=data_training,
                             max_tokens=MAX_TOKENS,
                             batch_size=BATCHES)

validation = (data_training.x_test, data_training.y_test)
model = SentimentCLF(nr_embed=data_training.vocab_size)
loss_func = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)

# training parameters
EPOCHS = 50
PROGRESS = 2

i = 1
for batch in batches:
    print('=================================')
    print(f'Batch [{i}]')
    i+=1
    j = 1
    for subbatch in batch:
        print(f'Sub_Batch [{j}]: {subbatch[0].size()}')
        j+= 1



def train(epochs):
    """
    for epoch in range(epochs):
        current_loss = 0.0
        for i, (inputs, label) in enumerate(data_training):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()

            if (i + 1) % PROGRESS == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Batch [{i + 1}/{len(data_training.x_train)}], Loss: {loss.item():.4f}")

            current_loss += loss.item

        epoch_loss = current_loss / len(data_training.x_train)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
        """
