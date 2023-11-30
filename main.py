import pandas as pd
import numpy as np
import torch as t
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from TorchNet import SimpleNN


def monk(path: str):
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

    # Load the dataset into a pandas DataFrame with specified data types
    train_data = pd.read_csv(train_file_path, names=column_names, sep=' ')
    test_data = pd.read_csv(test_file_path, names=column_names, sep=' ')

    # Display the first few rows of the loaded data

    train_data = train_data.drop(columns=['id'])
    test_data = test_data.drop(columns=['id'])

    numeric_columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    train_data[numeric_columns] = train_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    test_data[numeric_columns] = test_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # print(train_data.head())
    # print(test_data.head())

    # Convert data to PyTorch tensors (Train Data)
    train_features_tensor = t.Tensor(train_data[numeric_columns].values)
    train_labels_tensor = t.Tensor(train_data['class'].values)

    # For Test Data
    test_features_tensor = t.Tensor(test_data[numeric_columns].values)
    test_labels_tensor = t.Tensor(test_data['class'].values)

    net = SimpleNN(6, 6, 1)

    # Define the loss function
    loss_fn = t.nn.BCELoss()
    # SGD optimizer
    optimizer = t.optim.Adam(net.parameters(), lr=0.1)
    train_loss = []  # List to store losses for plotting
    test_loss = []  # List to store losses for plotting

    num_epochs = 50
    for epoch in range(num_epochs):

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(train_features_tensor)

        # Compute the loss
        loss = loss_fn(t.squeeze(outputs), train_labels_tensor)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss = loss.item()
        train_loss.append(running_loss)
        # Detach the outputs tensor before performing any operations on it
        outputs_detached = t.Tensor(np.round(outputs.detach().numpy()))
        # Compute accuracy using detached outputs
        acc = accuracy_score(train_labels_tensor.numpy(), t.squeeze(outputs_detached).round())
        # print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {running_loss}, Accuracy: {acc}")

            # Evaluate on test data
        with t.no_grad():
            test_outputs = net(test_features_tensor)
            val_loss = loss_fn(t.squeeze(test_outputs), test_labels_tensor)
            test_loss.append(val_loss.item())
    # plot loss
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

    # Evaluate on test data
    with t.no_grad():
        test_outputs = net(test_features_tensor)
        test_loss = loss_fn(t.squeeze(test_outputs), test_labels_tensor)
        print(
            f"Test Loss: {test_loss.item()}, Accuracy: {accuracy_score(test_labels_tensor, t.squeeze(test_outputs).round())})")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Replace 'path_to_train_file.train' with the actual path to your .train file
    monk("monks-1")
    monk("monks-2")
    monk("monks-3")
