import pandas as pd
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset


class CupDataset(Dataset):
    def __init__(self, dataframe):
        # Load the CSV file
        self.data = dataframe

        # Assuming the last three columns are the label and rest are features
        self.features = self.data.iloc[:, :-3].values
        self.labels = self.data.iloc[:, -3:].values

        # Convert data to PyTorch tensors
        self.features = t.tensor(self.features, dtype=t.float32)
        self.labels = t.tensor(self.labels, dtype=t.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(path: str):
    """
    Loads the dataset from a given path
        :param batch_size: mini-batch size
        :param folds: k-fold size
        :param path: to the csv file
        :return: list of train and validation loaders for each fold
    """
    # Load the dataset
    column_names = ['ID',
                    'INPUT_1', 'INPUT_2', 'INPUT_3', 'INPUT_4', 'INPUT_5', 'INPUT_6', 'INPUT_7', 'INPUT_8', 'INPUT_9',
                    'INPUT_10',
                    'TARGET_x', 'TARGET_y', 'TARGET_z']
    dataset = pd.read_csv(path, sep=',', comment='#', names=column_names, index_col='ID')

    return dataset


def torch_k_fold(batch_size, dataset, folds):
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=folds, shuffle=True)
    train_loaders = []
    val_loaders = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold_idx + 1}")
        # Create training and validation datasets for this fold
        train_dataset = Subset(CupDataset(dataset), train_idx)
        val_dataset = Subset(CupDataset(dataset), val_idx)

        # Create DataLoaders for training and validation
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size))
    return train_loaders, val_loaders


def skl_arange_dataset(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, scaler=None):
    X_dev = train_dataset.iloc[:, :-3].values
    y_dev = train_dataset.iloc[:, -3:].values

    X_test = test_dataset.iloc[:, :-3].values
    y_test = test_dataset.iloc[:, -3:].values

    if scaler is not None:
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)

    return X_dev, y_dev, X_test, y_test
