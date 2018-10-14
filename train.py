import torch
from config import config


def _train_iter(model, optimizer, criterion, batch):
    optimizer.zero_grad()
    X, Y = batch
    logic = model(X)  # logic: (batch,vocab_size,seq_len) Y:(batch,seq_len)
    loss = criterion(logic, Y)
    loss.backward()

    optimizer.step()

    return loss.item()


def training(model, optimizer, criterion, dataloader):
    model.train()
    loss = 0
    total_loss=0
    for i, batch in enumerate(dataloader,start=1):
        loss += _train_iter(model, optimizer, criterion, batch)
        if i % 50 == 0:
            loss=loss / 50
            print('loss :', loss)
            total_loss+=loss
            loss=0

    return total_loss / i
