# imports
from dataloader import *
from model import SentimentCLF
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from params import EPOCHS, LR, MAX_TOKENS, VOCAB_SIZE
import time


def data(load=False):
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
        train_loader = DynamicBatchLoader(dataset=data_train, max_tokens=MAX_TOKENS, debug=False).dynamic()
        eval_loader = DynamicBatchLoader(dataset=data_eval, max_tokens=MAX_TOKENS, debug=False).dynamic()
        torch.save(train_loader, 'train_batches.pt')
        torch.save(eval_loader, 'eval_batches.pt')
    else:
        pass


data(load=False)
batches = torch.load('train_batches.pt')
batches_eval = torch.load('eval_batches.pt')

model = SentimentCLF(nr_embed=VOCAB_SIZE)
loss_func = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)


def train(epochs):
    """
    Trains a pytorch model that
    stops training if after x
    epochs there ha
    """

    train_acc_epoch = []
    eval_acc_epoch = []
    train_loss_epoch = []
    eval_loss_epoch = []
    start = time.time()
    epoch_len = [x for x in range(1, epochs)]

    for epoch in range(epochs):
        current_loss = 0.0
        for i, batch in enumerate(batches):
            sequences, labels = batch[0], batch[1]

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

        print(f'================= Epoch {epoch + 1} =================')

        train_1, train_2 = accuracy(current_loss=current_loss, type_acc='train')
        train_acc_epoch.append(train_1)
        train_loss_epoch.append(train_2)

        eval_1, eval_2 = accuracy(current_loss=0, type_acc='eval')
        eval_acc_epoch.append(eval_1)
        eval_loss_epoch.append(eval_2)

    end = time.time()
    total = end - start
    minute = total / 60
    pool = model.pool

    torch.save(model.state_dict(), f'{pool}_{LR}_{MAX_TOKENS}.pt')
    print(f"Training time is {round(minute, 2)} minutes")


def accuracy(current_loss, type_acc='train'):
    truth = []
    pred = []
    with torch.no_grad():
        if type_acc == 'train':
            model.train()
            loss = current_loss / 20000
            for sequences, labels in batches:
                outputs = model(sequences)
                truth.append(labels)
                pred.append(torch.argmax(outputs, dim=1))

            truth = torch.cat(truth, dim=0)
            pred = torch.cat(pred, dim=0)
            acc = (truth == pred).float().mean().item()

            print(f"Training Loss: {loss:.4f}, Training Accuracy: {(acc * 100):.2f}%")

            return loss, acc

        elif type_acc == 'eval':
            model.eval()
            eval_loss = 0.0
            for sequences, labels in batches_eval:
                outputs = model(sequences)
                loss = loss_func(outputs, labels)
                eval_loss += loss.item()
                truth.append(labels)
                pred.append(torch.argmax(outputs, dim=1))
            truth = torch.cat(truth, dim=0)
            pred = torch.cat(pred, dim=0)
            acc = (truth == pred).float().mean().item()

            eval_loss /= 5000
            print(f"Validation Loss: {eval_loss:.4f}, Validation Accuracy: {(acc * 100):.2f}%")
            return loss, acc


train(epochs=EPOCHS)
