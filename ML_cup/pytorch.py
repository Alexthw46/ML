import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


def torch_train(model, train_loader: DataLoader, optimizer, epochs: int, loss_fn=MSELoss(), fold: int = 0,
                val_loader: DataLoader = None,
                scheduler: lr_scheduler = None, scheduler_on_val=True, verbose=True):
    train_loss = []
    val_loss = []
    avg_loss, avg_vloss = 0, 0
    if verbose:
        print(f'Fold: {fold}')
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_vloss = 0

        # minibatch training loop
        for (data, target) in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        avg_loss = epoch_loss / (len(train_loader))
        train_loss.append(avg_loss)
        if not (scheduler_on_val or scheduler is None):
            scheduler.step(metrics=avg_loss)

        if val_loader is not None:
            with torch.no_grad():
                for (data, target) in val_loader:
                    output = model(data)
                    loss = loss_fn(output, target)
                    epoch_vloss += loss.item()
            avg_vloss = epoch_vloss / len(val_loader)

            val_loss.append(avg_vloss)
            if scheduler_on_val and scheduler is not None:
                scheduler.step(metrics=avg_vloss)

        if verbose:
            print(f'Train Epoch: {epoch} Loss: {avg_loss}' + (
                '' if val_loader is None else f' Val Loss: {avg_vloss}'))
    if verbose:
        plot_loss(train_loss[10:], val_loss=val_loss[10:], fold=fold)

    return avg_loss, avg_vloss


class MEELoss(nn.Module):
    def __init__(self):
        super(MEELoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.norm(y_pred - y_true, dim=-1))


def MLP():
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(10, 50)),
        ('relu1', nn.Tanh()),
        ('fc2', nn.Linear(50, 25)),
        ('relu2', nn.Tanh()),
        ('fc3', nn.Linear(25, 25)),
        ('relu3', nn.Tanh()),
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


def grid_search(model_builder, parameters, train_loader, val_loader, scheduler, loss_fn=MSELoss(),
                max_epochs=100,
                verbose=True):
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_parameters = None

    for parameter_set in parameters:
        optimizer_type = parameter_set['optimizer']
        hyperparameters = {key: value for key, value in parameter_set.items() if key != 'optimizer'}

        # Generate all combinations of hyperparameters
        keys, values = zip(*hyperparameters.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for combo in combinations:

            train, val = grid_search_inner(model_builder, optimizer_params=(optimizer_type, combo),
                                           train_loader=train_loader,
                                           val_loader=val_loader,
                                           scheduler=scheduler,
                                           epochs=max_epochs,
                                           verbose=verbose)

            if val < best_val_loss:
                best_train_loss = train
                best_val_loss = val
                best_parameters = combo.copy()
                best_parameters['optimizer'] = optimizer_type
                print(
                    f'New Best Parameters: {best_parameters},Train Loss: {best_train_loss}, Val Loss: {best_val_loss}')

    print(
        f'Best Parameters: {best_parameters}, Train Loss: {best_train_loss}, Val Loss: {best_val_loss}')

    return best_parameters


def get_optimizer(optimizer_type, model, hyperparameters):
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), **hyperparameters)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(), **hyperparameters)


def get_scheduler(optimizer, scheduler_type, hyperparameters):
    if scheduler_type == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **hyperparameters)
    elif scheduler_type == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **hyperparameters)
    else:
        return None


def grid_search_inner(model_builder, optimizer_params, train_loader, val_loader, epochs, scheduler=('', None),
                      loss_fn=MSELoss(),
                      verbose=True):
    train_loss, val_loss = [], []  # Lists to store losses for each fold
    for train_loader, val_loader in zip(train_loader, val_loader):
        model = model_builder()
        optimizer = get_optimizer(optimizer_params[0], model, optimizer_params[1])
        # we ignore the scheduler for now
        # code for training the model with cross validation, returns the loss for last epoch of each fold
        train, val = torch_train(model, train_loader, loss_fn=loss_fn, optimizer=optimizer, val_loader=val_loader,
                                 epochs=epochs,
                                 scheduler=get_scheduler(optimizer, scheduler[0], scheduler[1]), verbose=verbose)
        train_loss.append(train)
        val_loss.append(val)

    return np.mean(train_loss), np.mean(val_loss)
