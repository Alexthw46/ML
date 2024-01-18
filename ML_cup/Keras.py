import itertools

import numpy as np
import tensorflow.keras as k
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras.backend import mean, sqrt, sum, square

from tensorflow.keras.optimizers import Adam, SGD, RMSprop


def mee(y_true, y_pred):
    # Calculate Euclidean distance between y_true and y_pred
    euclidean_distance = sqrt(sum(square(y_true - y_pred), axis=-1))
    # Calculate mean Euclidean error
    mean_euclidean_error = mean(euclidean_distance, axis=-1)
    return mean_euclidean_error


def keras_train(model: k.Model, optimizer, train_data, val_data, callback: list[callbacks.Callback],
                epochs=100, batch_size=32):
    model.compile(optimizer=optimizer, loss="mse", metrics=[mee], jit_compile=True)

    train_X, train_y = train_data
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=val_data,
                        callbacks=callback, shuffle=True)
    return history


def plot_keras_history(history, start_epoch=0):
    epoch_range = range(start_epoch, len(history.history['loss']))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot training and validation loss on the first subplot
    ax1.plot(epoch_range, history.history['loss'][start_epoch:], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(epoch_range, history.history['val_loss'][start_epoch:], label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot training and validation mee on the second subplot
    ax2.plot(epoch_range, history.history['mee'][start_epoch:], label='Training mee')
    if 'val_mee' in history.history:
        ax2.plot(epoch_range, history.history['val_mee'][start_epoch:], label='Validation mee')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mee')
    ax2.legend()

    # Set the title for the entire figure
    plt.suptitle('Training History')

    # Show the plot
    plt.show()


# method to create a model
def keras_mlp(hidden_layers, input_dim=10, output_dim=3, initializer=k.initializers.Orthogonal(seed=18)):
    input = layers.Input(shape=(input_dim,))
    x = input

    for name, layer_param in hidden_layers:
        if name == 'dense':
            x = layers.Dense(layer_param, activation='tanh', kernel_initializer=initializer)(x)
        elif name == 'dropout':
            x = layers.Dropout(layer_param)(x)
        elif name == 'bnorm':
            x = layers.BatchNormalization()(x)

    output = layers.Dense(output_dim, activation='linear', kernel_initializer=initializer)(x)
    return k.Model(input, output)


def keras_grid_search(model_builder, model_layers, parameters, train_data, val_data, max_epochs=100, verbose=0,
                      best_values=(20, 20)):
    best_train_loss, best_val_loss = best_values
    best_parameters = None

    for parameter_set in parameters:
        optimizer_type = parameter_set['optimizer']
        hyperparameters = {key: value for key, value in parameter_set.items() if key != 'optimizer'}

        # Generate all combinations of hyperparameters
        keys, values = zip(*hyperparameters.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for combo in combinations:
            print(f'Training with parameters: {combo}')
            train, val = keras_grid_search_inner(model_builder, model_layers, optimizer_type, train_data,
                                                 val_data,
                                                 epochs=max_epochs, verbose=verbose, params=combo)
            print(f'Train Loss: {train}, Val Loss: {val}')
            if train < best_train_loss and val < best_val_loss:
                best_train_loss = train
                best_val_loss = val
                best_parameters = combo.copy()
                best_parameters['optimizer'] = optimizer_type
                print('--------------------------------')
                print(f'New best parameters')
                print('--------------------------------')

    print(f'Best Parameters: {best_parameters}, Train Loss: {best_train_loss}, Val Loss: {best_val_loss}')
    return best_parameters, (best_train_loss, best_val_loss)


def keras_grid_search_inner(model_builder, model_layers, optimizer_type, train_data, val_data, epochs=100, verbose=0,
                            params=None):
    if params is None:
        params = {}
    train_loss, val_loss = [], []
    callback = [
        callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0, min_delta=1e-7),
        callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=10, cooldown=20, verbose=0,
                                    factor=0.31,
                                    min_lr=1e-7,
                                    min_delta=1e-7),
    ]
    # train all folds and get the average loss
    for train_dataset, val_dataset in zip(train_data, val_data):
        train_features, train_labels = train_dataset

        model: k.models.Model = model_builder(model_layers)
        if optimizer_type == 'Adam':
            optimizer = Adam(**params)
        elif optimizer_type == 'SGD':
            optimizer = SGD(**params)
        elif optimizer_type == 'RMSprop':
            optimizer = RMSprop(**params)
        else:
            raise ValueError('Optimizer not supported')

        model.compile(optimizer=optimizer, loss="mse", metrics=[mee], jit_compile=True)
        history = model.fit(train_features, train_labels, epochs=epochs, verbose=verbose,
                            validation_data=val_dataset, batch_size=32, callbacks=callback)

        train_loss.append(history.history['loss'][-1])
        val_loss.append(history.history['val_loss'][-1])

    return np.mean(train_loss), np.mean(val_loss)
