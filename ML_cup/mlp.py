from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


def train(model, train_loader: DataLoader, fold: int, loss_fn, optimizer, epochs: int, val_loader: DataLoader = None,
          scheduler: lr_scheduler = None):
    train_loss = []
    val_loss = []
    print(f'Fold: {fold}')
    for epoch in range(epochs):
        epoch_loss = 0
        # minibatch training loop
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            if batch_idx == len(train_loader) - 1:
                print(f'Train Epoch: {epoch} [{batch_idx + 1}/{len(train_loader)} Loss: {epoch_loss / (batch_idx + 1)}')
        train_loss.append(epoch_loss / len(train_loader))
        epoch_loss = 0

        if val_loader is not None:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    output = model(data)
                    loss = loss_fn(output, target)
                    epoch_loss += loss.item()
                    if batch_idx == len(val_loader) - 1:
                        print(f'Val Epoch: {epoch} [{batch_idx + 1}/{len(val_loader)}'
                              f' Loss: {epoch_loss / (batch_idx + 1)}')

            val_loss.append(epoch_loss / len(val_loader))
            if scheduler is not None:
                scheduler.step(metrics=epoch_loss)

    plot_loss(train_loss, val_loss=val_loss, fold=fold)


def MLP():
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(10, 50)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(50, 25)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(25, 25)),
        ('relu3', nn.ReLU()),
        ('final', nn.Linear(25, 3))
    ]))


def plot_loss(train_loss, fold: int, val_loss=None):
    # Plotting the loss curve
    plt.figure(figsize=(6, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for fold: {fold}')
    plt.legend()
    plt.grid(True)
    plt.show()
