import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Funzione per leggere i dataset MONK e trasformarli in DataFrame
def load_monk_data(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # divisione riga in parti separate da spazi
            if len(parts) > 1:
                label = int(parts[0])  # la prima colonna è il target
                features = list(map(int, parts[1:-1]))  # esclusione dell'ultima colonna (ID) e della prima colonna (target)
                data.append([label] + features)

    # Creazione del DataFrame con appositi nomi
    columns_names = ['Label'] + [f'Feature_{i}' for i in range(1, len(data[0]))]
    df = pd.DataFrame(data, columns=columns_names)
    return df

# Funzione per leggere i dataset MONK e trasformarli in DataFrame tramite codifica "get_dummies"
def load_monk_data_get_dummies(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # divisione riga in parti separate da spazi
            if len(parts) > 1:
                label = int(parts[0])  # la prima colonna è il target
                features = list(map(int, parts[1:-1]))  # esclusione dell'ultima colonna (ID) e della prima colonna (target)
                data.append([label] + features)

    # Creazione del DataFrame con appositi nomi
    columns_names = ['Label'] + [f'Feature_{i}' for i in range(1, len(data[0]))]
    df = pd.DataFrame(data, columns=columns_names)
    # Encoding delle colonne numeriche con "get_dummies"
    numeric_columns_names = ['Feature_1','Feature_2','Feature_3','Feature_4','Feature_5','Feature_6']
    df_encoded = pd.get_dummies(df, columns=numeric_columns_names)
    return df_encoded

# Funzione per leggere i dataset MONK e trasformarli in DataFrame tramite codifica "OneHotEncoder"
def load_monk_data_one_hot_enc(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # divisione riga in parti separate da spazi
            if len(parts) > 1:
                label = int(parts[0])  # la prima colonna è il target
                features = list(map(int, parts[1:-1]))  # esclusione dell'ultima colonna (ID) e della prima colonna (target)
                data.append([label] + features)

    # Creazione del DataFrame con nomi appositi nomi
    columns_names = ['Label'] + [f'Feature_{i}' for i in range(1, len(data[0]))]
    df = pd.DataFrame(data, columns=columns_names)
    # One-hot encoding delle colonne numeriche
    numeric_columns_names = ['Feature_1','Feature_2','Feature_3','Feature_4','Feature_5','Feature_6']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_np = encoder.fit_transform(df[numeric_columns_names])
    # Creazione del DataFrame con i dati codificati
    encoded_df = pd.DataFrame(encoded_np, columns=encoder.get_feature_names_out(numeric_columns_names))
    # Conversione dei valori float in interi (0 e 1)
    encoded_df = encoded_df.astype(int)
    # Aggiunta colonna 'Label' al DataFrame codificato
    encoded_df = pd.concat([df[['Label']], encoded_df], axis=1)
    return encoded_df

# Funzione per scalare i dati con MinMaxScaler
def StandardScalerFun(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[StandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    
    # Adatta solo sui dati di training
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
    
    # Trasforma i set di validazione e test con lo stesso scaler
    df_val_scaled = pd.DataFrame(scaler.transform(df_val), columns=df_train.columns, index=df_val.index)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_train.columns, index=df_test.index)

    return scaler, df_train_scaled, df_val_scaled, df_test_scaled

# Funzione per scalare i dati con MinMaxScaler
def MinMaxScalerFun(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    
    # Adatta solo sui dati di training
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
    
    # Trasforma i set di validazione e test con lo stesso scaler
    df_val_scaled = pd.DataFrame(scaler.transform(df_val), columns=df_train.columns, index=df_val.index)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_train.columns, index=df_test.index)

    return scaler, df_train_scaled, df_val_scaled, df_test_scaled

# Funzione che calcola il MEE (Mean Euclidian Error) (must be implemented by the user, and then set as a custom scoring function in model evaluation)
def mean_euclidean_error(y_true, y_pred) -> float:
    errors = y_true - y_pred
    return np.linalg.norm(errors, axis=1).mean()