from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


def train(model, train_loader, fold, loss_fn, optimizer, epochs, val_loader=None):
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
                        print(f'Val Epoch: {epoch} [{batch_idx + 1}/{len(val_loader)} Loss: {epoch_loss / (batch_idx + 1)}')
            val_loss.append(epoch_loss / len(val_loader))

    plot_loss(train_loss, val_loss=val_loss, fold=fold)


def MLP():
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(10, 50)),
        ('relu1', nn.Tanh()),
        ('fc2', nn.Linear(50, 50)),
        ('relu2', nn.Tanh()),
        ('fc3', nn.Linear(50, 3)),
        ('sigmoid', nn.Sigmoid())
    ]))


def plot_loss(train_loss, fold, val_loss=None):
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
