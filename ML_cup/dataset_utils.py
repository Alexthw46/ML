import pandas as pd
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset


class CupDataset(Dataset):
    def __init__(self, dataframe, device=t.device('cpu')):
        # Load the CSV file
        self.data = dataframe
        self.device = device
        # Assuming the last three columns are the label and rest are features
        self.features = self.data.iloc[:, :-3].values
        self.labels = self.data.iloc[:, -3:].values

        # Convert data to PyTorch tensors
        self.features = t.tensor(self.features, dtype=t.float32).to(self.device)
        self.labels = t.tensor(self.labels, dtype=t.float32).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_dataset(path: str):
    """
    Loads the dataset from a given path
        :param path: to the csv file
        :return: the dataset as a pandas dataframe
    """
    # Load the dataset
    column_names = ['ID',
                    'INPUT_1', 'INPUT_2', 'INPUT_3', 'INPUT_4', 'INPUT_5', 'INPUT_6', 'INPUT_7', 'INPUT_8', 'INPUT_9',
                    'INPUT_10',
                    'TARGET_x', 'TARGET_y', 'TARGET_z']
    dataset = pd.read_csv(path, sep=',', comment='#', names=column_names, index_col='ID', dtype=float)

    return dataset


def torch_split_dataset(dataset, train_ratio, batch_size):
    """
    :param dataset: dataframe to split
    :param train_ratio: percentage of training set
    :param batch_size: batch size
    :return: DataLoaders (CupDataset) for training and validation, with the given batch size
    """

    # Split the dataset into train and validation
    train_size = int(train_ratio * len(dataset))  # 80% train, 20% validation (adjust as needed)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    if t.cuda.is_available():
        device = t.device('cuda')
    else:
        device = t.device('cpu')
    # Create Subset datasets using CupDataset
    train_dataset = Subset(CupDataset(dataset, device), train_indices)
    val_dataset = Subset(CupDataset(dataset, device), val_indices)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=t.Generator(device=device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def torch_k_fold(batch_size, dataset, folds):
    """
    :param dataset: dataset to split
    :param batch_size: mini-batch size
    :param folds: k-fold size
    :return: list of train and validation loaders for each fold
    """
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=folds, shuffle=True)
    train_loaders = []
    val_loaders = []
    if t.cuda.is_available():
        device = t.device('cuda')
    else:
        device = t.device('cpu')

    for train_idx, val_idx in kf.split(dataset):
        # Create training and validation datasets for this fold
        train_dataset = Subset(CupDataset(dataset, device=device), train_idx)
        val_dataset = Subset(CupDataset(dataset, device=device), val_idx)

        # Create DataLoaders for training and validation
        train_loaders.append(
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=t.Generator(device=device)))
        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size))
    return train_loaders, val_loaders


def arange_datasets(train_dataset: pd.DataFrame, test_dataset: pd.DataFrame):
    X_dev = train_dataset.iloc[:, :-3].values
    y_dev = train_dataset.iloc[:, -3:].values

    X_test = test_dataset.iloc[:, :-3].values

    return X_dev, y_dev, X_test


def train_val_kfold(X_dev, y_dev, folds, random_state):
    # Lists to store KFold splits
    train_folds = []
    val_folds = []
    if folds > 1:
        # Initialize KFold
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)  # Adjust parameters as needed

        for train_index, val_index in kf.split(X_dev):
            train_features, val_features = X_dev[train_index], X_dev[val_index]
            train_target, val_target = y_dev[train_index], y_dev[val_index]

            train_folds.append((train_features, train_target))
            val_folds.append((val_features, val_target))

    return train_folds, val_folds
