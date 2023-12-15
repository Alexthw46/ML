import pandas as pd
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import torch.nn.functional
from sklearn.metrics import accuracy_score

from TorchNet import SimpleNN


def monk(path: str, optimizer: torch.optim.Optimizer,
         neural_network: torch.nn.Module = SimpleNN(17, 3, 1),
         num_epochs=50,
         lr_scheduler=None,
         eps=0.0005):
    test_features_tensor, test_labels_tensor, train_features_tensor, train_labels_tensor = load_dataset(path)

    # Define the loss function
    loss_fn = t.nn.MSELoss()

    train_loss = []  # List to store losses for plotting
    test_loss = []  # List to store losses for plotting
    # learning rate scheduler
    scheduler = lr_scheduler

    # early stopping
    patience = 7
    patience_counter = 0
    prev_loss = 1

    # Training loop
    for epoch in range(num_epochs):

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = neural_network(train_features_tensor)

        # Compute the loss
        loss = loss_fn(t.squeeze(outputs), train_labels_tensor)
        loss.backward()

        running_loss = loss.item()
        if prev_loss - running_loss < eps:
            patience_counter += 1
        else:
            patience_counter = 0
            prev_loss = running_loss
        if patience == patience_counter:
            break

        train_loss.append(running_loss)

        # Compute accuracy using predicted labels and train_labels_tensor
        acc = accuracy_score(train_labels_tensor.numpy(), t.round(outputs).detach().numpy())
        if epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {running_loss}, Accuracy: {acc}")

        # Evaluate on test data
        with t.no_grad():
            test_outputs = neural_network(test_features_tensor)
            val_loss = loss_fn(t.squeeze(test_outputs), test_labels_tensor)
            test_loss.append(val_loss.item())

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # Evaluate on test data
    with t.no_grad():
        test_outputs = neural_network(test_features_tensor)
        final_loss = loss_fn(t.squeeze(test_outputs), test_labels_tensor)
        acc = accuracy_score(test_labels_tensor.numpy(), t.round(test_outputs).detach().numpy())
        print(f"Test Loss: {final_loss.item()}, Accuracy: {acc}")

    # Plotting the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_dataset(path, verbose=False):
    train_file_path = 'data/monk+s+problems/' + path + '.train'
    test_file_path = 'data/monk+s+problems/' + path + '.test'
    column_names = [
        'class',
        'a1',
        'a2',
        'a3',
        'a4',
        'a5',
        'a6',
        'id'
    ]
    # Load the dataset into a pandas DataFrame
    train_data = pd.read_csv(train_file_path, names=column_names, sep=' ')
    test_data = pd.read_csv(test_file_path, names=column_names, sep=' ')
    train_data = train_data.drop(columns=['id'])
    test_data = test_data.drop(columns=['id'])
    numeric_columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    train_data = pd.get_dummies(train_data, columns=numeric_columns)
    test_data = pd.get_dummies(test_data, columns=numeric_columns)

    if verbose:
        print(train_data.head())

    # Convert output data to PyTorch tensors
    train_labels_tensor = t.Tensor(train_data['class'].values)
    test_labels_tensor = t.Tensor(test_data['class'].values)
    # Convert input data to PyTorch tensors
    train_features_tensor = t.Tensor(train_data.drop('class', axis=1).values)
    test_features_tensor = t.Tensor(test_data.drop('class', axis=1).values)
    return test_features_tensor, test_labels_tensor, train_features_tensor, train_labels_tensor
