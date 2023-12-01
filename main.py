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
         lr_scheduler=None):
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

    # Convert output data to PyTorch tensors
    train_labels_tensor = t.Tensor(train_data['class'].values)
    test_labels_tensor = t.Tensor(test_data['class'].values)

    # Convert input data to PyTorch tensors
    train_features_tensor = t.Tensor(train_data.drop('class', axis=1).values)
    test_features_tensor = t.Tensor(test_data.drop('class', axis=1).values)

    # Define the loss function
    loss_fn = t.nn.BCELoss()

    train_loss = []  # List to store losses for plotting
    test_loss = []  # List to store losses for plotting
    # learning rate scheduler
    scheduler = lr_scheduler

    # Training loop
    for epoch in range(num_epochs):

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = neural_network(train_features_tensor)

        # Compute the loss
        loss = loss_fn(t.squeeze(outputs), train_labels_tensor)
        loss.backward()

        # Print statistics
        running_loss = loss.item()
        train_loss.append(running_loss)
        # Detach the outputs tensor before performing any operations on it
        outputs_detached = t.Tensor(np.round(outputs.detach().numpy()))
        # Compute accuracy using detached outputs
        acc = accuracy_score(train_labels_tensor.numpy(), t.squeeze(outputs_detached).round())
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
        print(
            f"Test Loss: {final_loss.item()}, Accuracy: {accuracy_score(test_labels_tensor, t.squeeze(test_outputs).round())})")

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Replace 'path_to_train_file.train' with the actual path to your .train file
    monk("monks-1")
    monk("monks-2")
    monk("monks-3")
