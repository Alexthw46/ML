import tensorflow.keras as k
from keras import layers
from keras.src.utils.version_utils import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import numpy as np
import itertools

from tensorflow.keras.backend import mean, sqrt, sum, square
from tensorflow.keras.losses import Loss


def keras_train(model, optimizer, train_data, val_data, epochs=100, batch_size=32):
    model.compile(optimizer=optimizer, loss=k.losses.MeanSquaredError(), metrics=['accuracy'])
    callback = [
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=7, cooldown=2, verbose=1, factor=0.25,
                                    min_lr=1e-7,
                                    min_delta=1e-5),
        callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1, min_delta=1e-5,
                                restore_best_weights=True)
    ]
    train_X, train_y = train_data
    val_X, val_y = val_data
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y),
                        callbacks=callback)
    return history


# method to create a model
def keras_mlp(hidden_layers, input_dim=10, output_dim=3):
    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for layer_param, name in hidden_layers:
        if name == 'dense':
            model.add(layers.Dense(layer_param, activation='relu'))
        elif name == 'dropout':
            model.add(layers.Dropout(layer_param))
        elif name == 'bnorm':
            model.add(layers.BatchNormalization())
    model.add(layers.Dense(output_dim, activation='linear'))
    return model


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
            train, val = keras_grid_search_inner(model_builder, model_layers, optimizer_type, train_data, val_data,
                                                 epochs=max_epochs, verbose=verbose, params=combo)
            if val < best_val_loss:
                best_train_loss = train
                best_val_loss = val
                best_parameters = combo.copy()
                best_parameters['optimizer'] = optimizer_type
                print(
                    f'New Best Parameters: {best_parameters}, Train Loss: {best_train_loss}, Val Loss: {best_val_loss}')

    print(f'Best Parameters: {best_parameters}, Train Loss: {best_train_loss}, Val Loss: {best_val_loss}')
    return best_parameters, (best_train_loss, best_val_loss)


class MeanEuclideanError(Loss):
    def __init__(self, name='mean_euclidean_error'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return mean(sqrt(sum(square(y_pred - y_true), axis=-1)))


def keras_grid_search_inner(model_builder, model_layers, optimizer_type, train_data, val_data, epochs=100, verbose=0,
                            params=None):
    if params is None:
        params = {}
    train_loss, val_loss = [], []
    callback = [
        callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=10, cooldown=2, verbose=0, factor=0.25,
                                    min_lr=1e-7,
                                    min_delta=1e-5),
        callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, min_delta=1e-5,
                                restore_best_weights=True)
    ]
    loss = MeanEuclideanError()
    # train all folds and get the average loss
    for train_dataset, val_dataset in zip(train_data, val_data):
        train_features, train_labels = train_dataset

        model: k.models.Model = model_builder(model_layers)
        if optimizer_type == 'Adam':
            model.compile(optimizer=Adam(**params), loss=loss)
        elif optimizer_type == 'SGD':
            model.compile(optimizer=SGD(**params), loss=loss)
        elif optimizer_type == 'RMSprop':
            model.compile(optimizer=RMSprop(**params), loss=loss)
        else:
            raise ValueError('Optimizer not supported')

        history = model.fit(train_features, train_labels, epochs=epochs, verbose=verbose,
                            validation_data=val_dataset, batch_size=32, callbacks=callback)

        train_loss.append(history.history['loss'][-1])
        val_loss.append(history.history['val_loss'][-1])

    return np.mean(train_loss), np.mean(val_loss)
