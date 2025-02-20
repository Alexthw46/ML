import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame
from torch.nn import MSELoss
from torch.optim import lr_scheduler, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=1e-3):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.verbose = verbose
        self.delta = delta

    def check(self, val_loss):
        """Return True if early stopping condition is met (no improvement for `patience` consecutive epochs)."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss  # Update best loss
            self.counter = 0  # Reset patience counter
        else:
            self.counter += 1  # Increase patience counter

        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs of no improvement.")
            return True

        return False


def torch_train(model: torch.nn.Module, train_loader: DataLoader, optimizer: Optimizer, epochs: int, random_seed: int,
                loss_fn=MSELoss(),
                clip: float | None = 1.0,
                fold: int = 0,
                val_loader: DataLoader = None,
                scheduler: LRScheduler | None = None, scheduler_on_val=True, verbose=True, patience=10,
                skip_plot_points=20, return_last=True, writer: SummaryWriter = None) -> tuple[
                                                                                            np.floating, np.floating, np.floating, np.floating, np.floating] | \
                                                                                        tuple[
                                                                                            list[np.floating], list[
                                                                                                np.floating], list[
                                                                                                np.floating], list[
                                                                                                np.floating], np.floating]:
    torch.manual_seed(random_seed)
    train_loss = []
    val_loss = []
    mee_train, mee_val = [], []
    loss_fluctuation = []  # To track fluctuation in training loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    early_stopper = EarlyStopping(patience, verbose)  # Initialize early stopping
    mee = MEELoss()
    if verbose:
        print(f'Fold: {fold}')

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_mee_train = 0
        epoch_mee_val = 0

        model.train()
        # Minibatch training loop
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            epoch_loss += loss.item()
            epoch_mee_train += mee(output, target).item()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_loss)
        mee_train.append(epoch_mee_train / len(train_loader))

        # Calculate fluctuation (difference between consecutive losses)
        # we want to penalize models with high fluctuations but not rapid descent
        if len(train_loss) > 1:
            fluctuation = max(0, train_loss[-1] - train_loss[-2])
            loss_fluctuation.append(fluctuation)

        if not (scheduler_on_val or scheduler is None):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics=avg_loss)
            elif scheduler is not None:
                scheduler.step()

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                # minibatch validation loop
                for (data, target) in val_loader:
                    output = model(data)
                    loss = loss_fn(output, target)
                    epoch_mee_val += mee(output, target).item()
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            mee_val.append(epoch_mee_val / len(val_loader))
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics=avg_val_loss)
            elif scheduler_on_val and scheduler is not None:
                scheduler.step()

            if early_stopper.check(avg_val_loss if val_loader is not None else avg_loss):
                break  # Stop training if no improvement for `patience` epochs

        if verbose:
            print(f'Train Epoch: {epoch} Loss: {avg_loss}' +
                  ('' if val_loader is None
                   else f' Val Loss: {avg_val_loss}') +
                  ('' if scheduler is None
                   else (
                          ' lr: ' + str(scheduler.get_last_lr()))
                   )
                  )
        if writer is not None:
            writer.add_scalar(f'Loss/train/fold_{fold}', avg_loss, epoch)
            if val_loader is not None:
                writer.add_scalar(f'Loss/val/fold_{fold}', avg_val_loss, epoch)

    # Memory management: Clear model and optimizer
    del model
    del optimizer
    torch.cuda.empty_cache()  # If using GPU, clear GPU memory

    if verbose:
        plot_loss(train_loss, val_loss=val_loss, fold=fold)
        if skip_plot_points > 0:
            plot_loss(train_loss[skip_plot_points:], val_loss=val_loss[skip_plot_points:] if val_loss else [],
                      fold=fold,
                      skip=skip_plot_points)
        plot_mee(mee_train, mee_val)

    avg_fluctuation = np.mean(loss_fluctuation) if loss_fluctuation else 0
    if np.isnan(avg_fluctuation):
        print('Fluctuation was NaN')
        print(loss_fluctuation)
        avg_fluctuation = 0

    # Return the last loss values or the entire list
    if return_last:
        return train_loss[-1], val_loss[-1] if len(val_loss) > 0 else 0, mee_train[-1], mee_val[-1] if len(
            mee_val) > 0 else 0, avg_fluctuation
    else:
        return train_loss, mee_train, val_loss, mee_val, avg_fluctuation


def torch_predict(model, test_loader: DataLoader):
    # evaluate the model on the test set using mee metric
    predictions = []
    mee = []
    with torch.no_grad():
        for data, true in test_loader:
            output = model(data)
            predictions.append(output)
            mee.append(torch.norm(output - true, dim=-1))
    return torch.mean(torch.cat(mee)).item()


class MEELoss(nn.Module):
    def __init__(self):
        super(MEELoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.norm(y_pred - y_true, dim=-1))


# def MLP():
#     return nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(12, 250)),
#         ('relu1', nn.Tanh()),
#         ('fc2', nn.Linear(250, 300)),
#         ('relu2', nn.Tanh()),
#         ('fc3', nn.Linear(300, 350)),
#         ('relu3', nn.Tanh()),
#         ('final', nn.Linear(350, 3))
#     ]))
#
#
# def MLPv2():
#     return nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(12, 300)),
#         ('relu1', nn.Tanh()),
#         ('bn_1', nn.BatchNorm1d(300)),
#         ('fc2', nn.Linear(300, 250)),
#         ('bn_2', nn.BatchNorm1d(250)),
#         ('relu2', nn.Tanh()),
#         ('fc3', nn.Linear(250, 250)),
#         ('bn_3', nn.BatchNorm1d(250)),
#         ('relu3', nn.Tanh()),
#         ('final', nn.Linear(250, 3))
#     ]))
#
#
# def MLPv3():
#     return nn.Sequential(OrderedDict([
#         ('fc1', nn.Linear(12, 300)),
#         ('relu1', nn.Tanh()),
#         ('fc2', nn.Linear(300, 300)),
#         ('relu2', nn.Tanh()),
#         ('fc3', nn.Linear(300, 250)),
#         ('relu3', nn.Tanh()),
#         ('fc4', nn.Linear(250, 250)),
#         ('relu4', nn.Tanh()),
#         ('final', nn.Linear(250, 3))
#     ]))


def plot_loss(train_loss, fold: int, val_loss=None, skip=0):
    # Plotting the loss curve
    plt.figure(figsize=(6, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if skip > 0:
        plt.yscale('log')
        plt.title(f'Loss Curve for fold {fold} after {skip} epochs')
    else:
        plt.title(f'Loss Curve for fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mee(mee_train, mee_val):
    # Plotting the loss curve
    plt.figure(figsize=(6, 6))
    plt.plot(mee_train, label='Training MEE', color='blue')
    plt.plot(mee_val, label='Validation MEE', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('MEE')
    plt.title('MEE Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


# Parallelized grid search main function with memory management and refined reporting
def grid_search(model_builder: Callable[[], nn.Module], parameters: dict, train_loader: list[DataLoader],
                val_loader: list[DataLoader], scheduler: (str, dict), max_epochs: int = 100, random_seed: int = 42,
                stability_threshold: float = 0.1, patience: int = 10, clip: float = 1.0,
                tensorboard_folder_base: str = None) -> DataFrame | Any:
    results = []  # To store all results for reporting
    tensorboard_folder = tensorboard_folder_base
    # Generate hyperparameter combinations
    for parameter_set in parameters:
        optimizer_type = parameter_set['optimizer']
        hyperparameters = {key: value for key, value in parameter_set.items() if key != 'optimizer'}

        keys, values = zip(*hyperparameters.items())
        combinations = itertools.product(*values)

        # Parallel grid search using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = []
            for combo in combinations:
                if tensorboard_folder is not None:
                    tensorboard_folder = f'{tensorboard_folder_base}/{optimizer_type}_{combo}'

                futures.append(
                    executor.submit(grid_search_inner, model_builder,
                                    (optimizer_type, dict(zip(keys, combo)), clip),
                                    train_loader, val_loader, max_epochs, random_seed, scheduler, MSELoss(), patience,
                                    tensorboard_folder, stability_threshold))

            # Gather results
            for future in futures:
                train, val, fluctuation = future.result()
                param_dict = dict(zip(keys, combo))
                param_dict['optimizer'] = optimizer_type
                results.append((param_dict, train, val, fluctuation))

    # Convert results to DataFrame for easy sorting and reporting
    results_df = pd.DataFrame(results, columns=["param_set", "train_loss", "val_loss", "fluctuation"])
    results_df.to_csv(tensorboard_folder_base + '/results.csv')
    # Filter out with stability threshold before sorting
    results_df = results_df[results_df['fluctuation'] < stability_threshold]
    # Find the best configuration based on validation loss and stability
    best_result = results_df.sort_values(by=["val_loss", "fluctuation"], ascending=[True, True])
    # check if there are valid results before accessing the first element
    if not best_result.empty:
        best_result = best_result.iloc[0]
    else:
        return results_df
    best_parameters = best_result["param_set"]
    best_train_loss = best_result["train_loss"]
    best_val_loss = best_result["val_loss"]
    best_fluctuation = best_result["fluctuation"]

    print(
        f'Best Parameters: {best_parameters}, Train Loss: {best_train_loss}, Val Loss: {best_val_loss}, Stability: {best_fluctuation}')
    return best_parameters


def get_optimizer(optimizer_type, model, hyperparameters):
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), **hyperparameters)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(), **hyperparameters)
    else:
        print('Invalid optimizer type', optimizer_type)
        return None


# Works only with ReduceLROnPlateau, needs adjustment for other schedulers due to different hyperparameter formats
def get_scheduler(optimizer: Optimizer, scheduler_type: str, hyperparameters: dict) -> LRScheduler | None:
    if scheduler_type == 'StepLR':
        return lr_scheduler.StepLR(optimizer, **hyperparameters)
    elif scheduler_type == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **hyperparameters)
    elif scheduler_type == 'OneCycleLR':
        return lr_scheduler.OneCycleLR(optimizer, **hyperparameters)
    else:
        return None


# Parallelized grid search inner function that does the k-fold cross validation
def grid_search_inner(model_builder: Callable[[], nn.Module], optimizer_params: (str, dict, float),
                      train_loader: list[DataLoader],
                      val_loader: list[DataLoader], epochs: int, seed: int, scheduler=('', None),
                      loss_fn=MSELoss(),
                      patience: int = 10, tensorboard_folder: str = None, stability_threshold=0.1) -> tuple[
    np.floating, np.floating, np.floating]:
    writer = None
    if tensorboard_folder is not None:
        writer = SummaryWriter(tensorboard_folder)
    train_metric, validation_metric = [], []
    fluctuation_metrics = []  # To store fluctuation for each fold
    fold = 0
    for train_loader, val_loader in zip(train_loader, val_loader):
        model = model_builder()
        optimizer = get_optimizer(optimizer_params[0], model, optimizer_params[1])

        train, mee_t, val, mee_v, fluctuation = torch_train(model, train_loader, random_seed=seed, loss_fn=loss_fn,
                                                            optimizer=optimizer,
                                                            val_loader=val_loader, epochs=epochs,
                                                            scheduler=get_scheduler(optimizer, scheduler[0],
                                                                                    scheduler[1]),
                                                            verbose=False, patience=patience, fold=fold,
                                                            clip=optimizer_params[2],
                                                            writer=writer)
        train_metric.append(mee_t)
        validation_metric.append(mee_v)
        fluctuation_metrics.append(fluctuation)
        fold += 1

    avg_train_loss = np.mean(train_metric)
    avg_val_loss = np.mean(validation_metric)
    avg_fluctuation = np.mean(fluctuation_metrics)

    if writer is not None:
        writer.close()

    if avg_fluctuation < stability_threshold:
        print(f'Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}, Avg Fluctuation: {avg_fluctuation}')
        print(optimizer_params)
    return avg_train_loss, avg_val_loss, avg_fluctuation
