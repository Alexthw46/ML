from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error, root_mean_squared_error

#MEE
def mean_euclidean_error(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred))


mee_scorer = make_scorer(mean_euclidean_error, greater_is_better=False)


# Caricamento del dataset ML-CUP24
def load_cup_data(file_path):
    data = pd.read_csv(file_path, sep=",", comment="#", header=None)
    X = data.iloc[:, 1:13].values  # 12 feature numeriche
    y = data.iloc[:, 13:].values   # 3 target (x, y, z)
    return X, y

# Percorso del file di training
dataset_path = "../data/ML-CUP24-TR.csv"


# Carica i dati
X, y = load_cup_data(dataset_path)

scaler_input = StandardScaler()


#scaling
scaler = { 
    'TARGET_x': MinMaxScaler(),
    'TARGET_y': MinMaxScaler(),
    'TARGET_z': MinMaxScaler()
}

MinMax_values = {
    'TARGET_x': {'min':0, 'max':0},
    'TARGET_y': {'min':0, 'max':0},
    'TARGET_z': {'min':0, 'max':0}
}

MinMax_values['TARGET_x']['min'] = min(y[:, 0])
MinMax_values['TARGET_x']['max'] = max(y[:, 0])
MinMax_values['TARGET_y']['min'] = min(y[:, 1])
MinMax_values['TARGET_y']['max'] = max(y[:, 1])
MinMax_values['TARGET_z']['min'] = min(y[:, 2])
MinMax_values['TARGET_z']['max'] = max(y[:, 2])

def manual_inverse_MinMax(y, target):
    min = MinMax_values[target]['min']
    max = MinMax_values[target]['max']
    y_descaled = np.array(list(map(lambda x: (x * (max - min)) + min, y)))
    return y_descaled

#descaling delle predizioni
def predict_scaled_IO_training(model, x, target):
    pred_scaled = model.predict(scaler_input.transform(x))
    pred_scaled = pred_scaled.reshape(-1, 1)   
    pred_original = manual_inverse_MinMax(pred_scaled, target).reshape(-1)
    return pred_original

def predict_scaled_IO_training_on_scaled_data(model, x, target):
    pred_scaled = model.predict(x)
    pred_scaled = pred_scaled.reshape(-1, 1)  
    pred_original = manual_inverse_MinMax(pred_scaled, target).reshape(-1)
    return pred_original


# Scaling dei dati (normalizzazione tra 0 e 1)
X_scaled = scaler_input.fit_transform(X)


y_scaled = np.column_stack([
    scaler['TARGET_x'].fit_transform(y[:, 0].reshape(-1, 1)),  # Scaling per TARGET_x    
    scaler['TARGET_y'].fit_transform(y[:, 1].reshape(-1, 1)),  # Scaling per TARGET_y
    scaler['TARGET_z'].fit_transform(y[:, 2].reshape(-1, 1))   # Scaling per TARGET_z
    
])


# Suddivisione in training e test scalati
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


y_train_descaled = np.column_stack([
    manual_inverse_MinMax(y_train[:, 0].reshape(-1, 1), 'TARGET_x'),  # Scaling per TARGET_x
    manual_inverse_MinMax(y_train[:, 1].reshape(-1, 1), 'TARGET_y'),  # Scaling per TARGET_y
    manual_inverse_MinMax(y_train[:, 2].reshape(-1, 1), 'TARGET_z')   # Scaling per TARGET_z
])

y_test_descaled = np.column_stack([
    manual_inverse_MinMax(y_test[:, 0].reshape(-1, 1), 'TARGET_x'),  # Scaling per TARGET_x
    manual_inverse_MinMax(y_test[:, 1].reshape(-1, 1), 'TARGET_y'),  # Scaling per TARGET_y
    manual_inverse_MinMax(y_test[:, 2].reshape(-1, 1), 'TARGET_z')   # Scaling per TARGET_z
])


#Griglie per kernel con parametri variabili

# Griglia dei parametri per 'linear'
param_grid_linear = {
    'kernel': ['linear'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Griglia dei parametri per 'poly'
param_grid_poly = {
    'kernel': ['poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],
    'coef0': [0, 1, 10]
}

# Griglia dei parametri per 'rbf'
param_grid_rbf = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Griglia dei parametri per 'sigmoid'
param_grid_sigmoid = {
    'kernel': ['sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'coef0': [0, 1, 10]
}

# Combinare tutte le griglie in una lista
grids_svr = [param_grid_linear, param_grid_rbf, param_grid_poly, param_grid_sigmoid]

# Definizione degli iperparametri per XGBoost
grid_xgb = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'eval_metric': ['rmse'],
    'early_stopping_rounds': [5, 10, 15, 20]
}

evals_result = {
        "TARGET_x": {}, 
        "TARGET_y": {}, 
        "TARGET_z": {}
}


# Funzione per allenare e valutare un modello versione kernel con parametri variabili
def train_and_evaluate(model, grid, model_name):
    scoring = {
        'mse': 'neg_mean_squared_error',  # MSE negativo (perché sklearn massimizza di default)
        'mee': mee_scorer
    }
    results = {}
    best_models = {}
    best_params = {}
    y_pred_descaled = {}
    y_pred_train_descaled = {}
    y_pred = {}
    y_pred_train = {}
    for i, target_name in enumerate(["TARGET_x", "TARGET_y", "TARGET_z"]):
        
        if(model_name == "XGBoost"):
            grid_search = GridSearchCV(model, grid, cv=5, scoring=scoring, refit='mse', return_train_score=True)
            grid_search.fit(X_train, y_train[:, i],
                            **{ 
                                "eval_set": [(X_train, y_train[:, i]), (X_test, y_test[:, i])],
                                "verbose": False
                            })
            best_model = grid_search.best_estimator_
            evals_result[target_name] = best_model.evals_result()
        else: 
            best_model = None
            best_score = float('-inf')

            for g in grid:
                grid_search = GridSearchCV(model, g, cv=5, scoring=scoring, refit='mse', return_train_score=True)
                grid_search.fit(X_train, y_train[:, i])
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_

        cv_results = grid_search.cv_results_
        best_index = grid_search.best_index_ 


        mean_mse_scores = -cv_results['mean_test_mse'][best_index]  # Media MSE sui validation set
        mean_mee_scores = -cv_results['mean_test_mee'][best_index]  # Media MEE sui validation set
        mean_train_mse = -cv_results['mean_train_mse'][best_index]  # Media MSE sui training set
        mean_train_mee = -cv_results['mean_train_mee'][best_index]  # Media MEE sui training set
               
        
        y_pred_train[target_name] = best_model.predict(X_train)
        y_pred_train_descaled[target_name] = predict_scaled_IO_training_on_scaled_data(best_model, X_train, target_name)
        
        train_rmse = root_mean_squared_error(y_train[:, i], y_pred_train[target_name])
        train_mee = mean_euclidean_error(y_train[:, i], y_pred_train[target_name])
        train_rmse_descaled = root_mean_squared_error(scaler[target_name].inverse_transform(y_train[:, i].reshape(-1, 1)), y_pred_train_descaled[target_name])
        train_mee_descaled = mean_euclidean_error(scaler[target_name].inverse_transform(y_train[:, i].reshape(-1, 1)), y_pred_train_descaled[target_name])
        
 
        y_pred[target_name] = best_model.predict(X_test)
        y_pred_descaled[target_name] = predict_scaled_IO_training_on_scaled_data(best_model, X_test, target_name)
        
        rmse = root_mean_squared_error(y_test[:, i], y_pred[target_name])
        mee = mean_euclidean_error(y_test[:, i], y_pred[target_name])
        rmse_descaled = root_mean_squared_error(scaler[target_name].inverse_transform(y_test[:, i].reshape(-1, 1)), y_pred_descaled[target_name])
        mee_descaled = mean_euclidean_error(scaler[target_name].inverse_transform(y_test[:, i].reshape(-1, 1)), y_pred_descaled[target_name])

 
        results[target_name] = rmse
        best_models[target_name] = best_model
        best_params[target_name] = grid_search.best_params_

        print(f"{model_name} Best Params for {target_name}: {grid_search.best_params_}")

        print(f"{model_name} Validation in CV on scaled data for {target_name}: MEAN MSE={mean_mse_scores}, MEAN MEE={mean_mee_scores}")
        print(f"{model_name} Validation in CV on de scaled data estimated for {target_name}: MEAN MSE={mean_mse_scores * (MinMax_values[target_name]['max']- MinMax_values[target_name]['min'])}, MEAN MEE={mean_mee_scores * (MinMax_values[target_name]['max']- MinMax_values[target_name]['min'])}")
        print(f"{model_name} Training in CV on scaled data for {target_name}: MEAN MSE={mean_train_mse}, MEAN MEE={mean_train_mee}")
        print(f"{model_name} Training on retrained model for {target_name}: RMSE={train_rmse}, MEE={train_mee}")
        print(f"{model_name} Training on retrained model on scaled data for {target_name}: RMSE={train_rmse_descaled}, MEE={train_mee_descaled}")
        print(f"{model_name} Test on retrained model for {target_name}: RMSE={rmse}, MEE={mee}")
        print(f"{model_name} Test on retrained model on scaled data for {target_name}: RMSE={rmse_descaled}, MEE={mee_descaled}")

        
    y_pred_combined_test = np.column_stack((y_pred["TARGET_x"], y_pred["TARGET_y"], y_pred["TARGET_z"]))
    y_pred_combined_train = np.column_stack((y_pred_train["TARGET_x"], y_pred_train["TARGET_y"], y_pred_train["TARGET_z"]))
    y_pred_combined_test_descaled = np.column_stack((y_pred_descaled["TARGET_x"], y_pred_descaled["TARGET_y"], y_pred_descaled["TARGET_z"]))
    y_pred_combined_train_descaled = np.column_stack((y_pred_train_descaled["TARGET_x"], y_pred_train_descaled["TARGET_y"], y_pred_train_descaled["TARGET_z"]))

    rmse_test = root_mean_squared_error(y_pred_combined_test, y_test)
    rmse_train = root_mean_squared_error(y_pred_combined_train, y_train)
    rmse_test_desc = root_mean_squared_error(y_pred_combined_test_descaled, y_test_descaled)
    rmse_train_desc = root_mean_squared_error(y_pred_combined_train_descaled, y_train_descaled)

    print(f"RMSE of XGBoost complete model on: Test={rmse_test}, Train={rmse_train}, Test descaled={rmse_test_desc} ,Train descaled={rmse_train_desc} ")

    mee_test = mean_euclidean_error(y_pred_combined_test, y_test)
    mee_train = mean_euclidean_error(y_pred_combined_train, y_train)
    mee_test_desc = mean_euclidean_error(y_pred_combined_test_descaled, y_test_descaled)
    mee_train_desc = mean_euclidean_error(y_pred_combined_train_descaled, y_train_descaled)

    print(f"MEE of XGBoost complete model on: Test={mee_test}, Train={mee_train}, Test descaled={mee_test_desc}, Train descaled={mee_train_desc} ")

    return results, best_models, best_params

# Allenamento e valutazione per SVR
def svr_fit():
    svr_results, best_models, best_params = train_and_evaluate(SVR(), grids_svr, "SVR")
    return svr_results, best_models, best_params

# Allenamento e valutazione per XGBoost
def xgb_fit():
    xgb_results, best_models, best_params = train_and_evaluate(XGBRegressor(), grid_xgb, "XGBoost")
    return xgb_results, best_models, best_params, evals_result 



# Minibatch Training
def variable_batch_training_XGBoost( model_params, batch_size=32, epochs=10):
    minibatch_models = {}
    evals_result = {}
    train_rmse = {target: [] for target in model_params.keys()}
    test_rmse = {target: [] for target in model_params.keys()}

    for idx, target in enumerate(["TARGET_x", "TARGET_y", "TARGET_z"]):
        params = model_params[target]
        minibatch_models[target] = XGBRegressor(**params)
        
        for epoch in range(epochs):
            X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train[:, idx], random_state=42)
            for i in range(0, len(X_train_shuffled), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size].ravel()
                if epoch == 0 and i == 0:
                    minibatch_models[target].fit(X_batch, y_batch, eval_set=[(X_train, y_train[:, idx].reshape(-1, 1)), (X_test, y_test[:, idx].reshape(-1, 1))], verbose=False)
                else:    
                    minibatch_models[target].fit(X_batch, y_batch, xgb_model=minibatch_models[target].get_booster(), eval_set=[(X_train, y_train[:, idx].reshape(-1, 1)), (X_test, y_test[:, idx].reshape(-1, 1))], verbose=False)
                evals_result = minibatch_models[target].evals_result()
                train_rmse[target].extend(evals_result['validation_0']['rmse'])
                test_rmse[target].extend(evals_result['validation_1']['rmse'])

    return minibatch_models, train_rmse, test_rmse


