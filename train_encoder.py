# imports
from dataloader import *
from base_model import SentimentCLF
from encoder_model import EncoderModel
import torch
from torch.optim import SGD, Adam, Adamax
from torch.nn import CrossEntropyLoss
from params import *
from batch_functions import data2df, get_device
import time
import sys
import wandb

device = get_device()


def data(load=False):
    """""
    Call this function if you want to regenerate
    the same 'batches.pt' file locally.
    In this way you don't need to
    rerun this function everytime
    the script is ran.
    """""
    # Define training and eval/test data
    if load:
        data_train, data_eval = data2df(final=False)
        data_train, data_eval = DatasetSentiment(data_train), DatasetSentiment(data_eval)
        train_loader = DynamicBatchLoader(dataset=data_train, max_tokens=MAX_TOKENS, debug=False).dynamic()
        eval_loader = DynamicBatchLoader(dataset=data_eval, max_tokens=MAX_TOKENS, debug=False).dynamic()
        torch.save(train_loader, 'data/train_batches.pt')
        torch.save(eval_loader, 'data/eval_batches.pt')
    else:
        pass


data(load=False)
batches = torch.load('data/train_batches.pt')
batches_eval = torch.load('data/eval_batches.pt')


def train(epochs=EPOCHS,
          pool_type=POOL,
          lr=LR,
          embed_dim=EMBED_DIM,
          heads=HEADS,
          depth=ENCDEPTH,
          hidden=HIDDEN,
          dropout=DROPOUT):
    """""
    Trains a pytorch model that
    stops training if after x
    epochs there ha
    """""
    start = time.time()

    model = EncoderModel(embed_dim=embed_dim,
                         depth=depth,
                         hidden=hidden,
                         dropout=dropout,
                         pool_type=pool_type,
                         heads=heads)

    loss_func = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.to(device)

    # compute training and validation accuracy/loss
    def accuracy(current_loss, type_acc='train'):
        truth = []
        pred = []
        with torch.no_grad():
            if type_acc == 'train':
                model.train()
                loss_acc = current_loss / 20000
                for sequences, labels in batches:
                    outputs = model(sequences.to(device))
                    truth.append(labels.to(device))
                    pred.append(torch.argmax(outputs, dim=1))

                truth = torch.cat(truth, dim=0)
                pred = torch.cat(pred, dim=0)
                acc = (truth == pred).float().mean().item()
                acc_wandb = acc * 100
                wandb.log({"training_loss": loss_acc, "training_accuracy": acc_wandb})
                print(f"Training Loss: {loss_acc:.4f}, Training Accuracy: {(acc * 100):.2f}%")

                return acc, loss_acc

            elif type_acc == 'eval':
                model.eval()
                eval_loss = 0.0
                for sequences, labels in batches_eval:
                    outputs = model(sequences.to(device))
                    loss_acc = loss_func(outputs, labels.to(device))
                    eval_loss += loss_acc.item()
                    truth.append(labels.to(device))
                    pred.append(torch.argmax(outputs, dim=1))
                truth = torch.cat(truth, dim=0)
                pred = torch.cat(pred, dim=0)
                acc = (truth == pred).float().mean().item()

                eval_loss /= 5000
                eval_acc_wandb = acc * 100
                wandb.log({"validation_loss": eval_loss, "validation_accuracy": eval_acc_wandb})
                print(f"Validation Loss: {eval_loss:.4f}, Validation Accuracy: {(acc * 100):.2f}%")
                return acc, eval_loss

    batch_len = len(batches)

    # train for each epoch
    for epoch in range(epochs):
        model.train()
        print(f'================= Epoch {epoch + 1} =================')

        current_loss = 0.0
        # iterate through each batch
        for i, batch in enumerate(batches):
            percentage = round(((i / batch_len) * 100), 0)
            sys.stdout.write("\r================= {0}% ===================".format(str(percentage)))
            sys.stdout.flush()
            # define sequences and labels
            sequences, labels = batch[0].to(device), batch[1].to(device)

            # standard training calls
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = loss_func(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # update the loss
            current_loss += loss.item()

        print('\n')
        # save accuracy and loss in wandb project
        accuracy(current_loss=current_loss, type_acc='train')
        accuracy(current_loss=0, type_acc='eval')

    # compute training time
    end = time.time()
    total = end - start
    minute = total / 60

    # saving the model
    torch.save(model.state_dict(), f'models/EMB_{EMBED_DIM}_LR_{LR}_HEADS_{HEADS}_ENC_{ENCDEPTH}.pt')

    # printing training time
    print(f"Training time is {round(minute, 2)} minutes")


wandb.init(
    # set the wandb project where this run will be logged
    project=f"IMDB_Transformer_ENC_V3",
    name=f"EMB_{EMBED_DIM}_LR_{LR}_HEADS_{HEADS}_ENC_{ENCDEPTH}",

    # track hyper-parameters and run metadata
    config={
        "learning_rate": LR,
        "architecture": "Transformer",
        "dataset": "IMDB50000",
        "epochs": EPOCHS,
        "pooling": POOL,
        "attention": ATTENTION,
        "embedding": EMBED_DIM,
        "heads": HEADS,
        "depth": ENCDEPTH

    }
)

"""""
Training loop for a Encoder
classification model for 
sentiment analysis.
"""""
train(epochs=EPOCHS,
      pool_type=POOL,
      lr=LR,
      embed_dim=EMBED_DIM,
      heads=HEADS,
      depth=ENCDEPTH
      )
