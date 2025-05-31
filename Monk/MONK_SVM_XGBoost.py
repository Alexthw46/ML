from cmath import inf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import itertools
import os


def monk_project(n):
    # Caricamento del dataset MONK
    def load_monk_data(file_path):
        data = []
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                labels.append(int(parts[0]))  # Prima colonna è il target
                data.append([int(x) for x in parts[1:7]])  # Sei feature categoriali
        
        return pd.DataFrame(data), np.array(labels)

    # Percorso dei file di training e test
    dataset_path = "../data/monk+s+problems"
    train_file = os.path.join(dataset_path, f"monks-{n}.train")
    test_file = os.path.join(dataset_path, f"monks-{n}.test")

    # Carica i dati
    X_train_full, y_train_full = load_monk_data(train_file)
    X_test, y_test = load_monk_data(test_file)

    # One-Hot Encoding per variabili categoriche
    encoder = OneHotEncoder()
    X_train_encoded_full = encoder.fit_transform(X_train_full).toarray()
    X_test_encoded = encoder.transform(X_test).toarray()

    k = 5  # Numero di fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


    # Definizione degli iperparametri per la ricerca manuale

    grids_svm = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['poly'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'degree': [2, 3], 'coef0': [0, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'coef0': [0, 1, 10]}
    ]

    # Testiamo tutte le combinazioni di iperparametri per SVM
    best_avg_acc_svm=0
    svm_tr_acc_ba = 0
    svm_ts_acc_ba = 0
    svm_tr_mse_ba = 0
    svm_ts_mse_ba = 0
    best_params_svm_acc={}
    y_pred_best_svm_acc=[]

    best_avg_mse_svm=inf
    svm_tr_acc_br = 0
    svm_ts_acc_br = 0
    svm_tr_mse_br = 0
    svm_ts_mse_br = 0
    best_params_svm_mse={}
    y_pred_best_svm_mse=[]

    for grid_svm in grids_svm:
        for params in itertools.product(*grid_svm.values()):

            avg_acc=0
            avg_mse=0

            param_dict = dict(zip(grid_svm.keys(), params))

            acc_scores = []
            mse_scores = []
            
            for train_index, val_index in skf.split(X_train_encoded_full, y_train_full):
                X_train, X_val = X_train_encoded_full[train_index], X_train_encoded_full[val_index]
                y_train, y_val = y_train_full[train_index], y_train_full[val_index]

                svm = SVC(**param_dict)
                svm.fit(X_train, y_train)
                y_pred_svm = svm.predict(X_val)

                acc_scores.append(accuracy_score(y_val, y_pred_svm))
                mse_scores.append(mean_squared_error(y_val, y_pred_svm))

            
            avg_acc = np.mean(acc_scores)
            if(avg_acc>best_avg_acc_svm):

                #retraining
                svm.fit(X_train_encoded_full, y_train_full)

                y_pred_test_svm = svm.predict(X_test_encoded)
                y_pred_train_svm = svm.predict(X_train_encoded_full)

                svm_tr_acc_ba = accuracy_score(y_train_full, y_pred_train_svm)
                svm_ts_acc_ba = accuracy_score(y_test, y_pred_test_svm)
                svm_tr_mse_ba = mean_squared_error(y_train_full, y_pred_train_svm)
                svm_ts_mse_ba = mean_squared_error(y_test, y_pred_test_svm)

                best_avg_acc_svm = avg_acc
                best_params_svm_acc = param_dict
                y_pred_best_svm_acc = y_pred_test_svm
            
            avg_mse = np.mean(mse_scores)
            if(avg_mse<best_avg_mse_svm):

                #retraining
                svm.fit(X_train_encoded_full, y_train_full)

                y_pred_test_svm = svm.predict(X_test_encoded)
                y_pred_train_svm = svm.predict(X_train_encoded_full)

                svm_tr_acc_br = accuracy_score(y_train_full, y_pred_train_svm)
                svm_ts_acc_br = accuracy_score(y_test, y_pred_test_svm)
                svm_tr_mse_br = mean_squared_error(y_train_full, y_pred_train_svm)
                svm_ts_mse_br = mean_squared_error(y_test, y_pred_test_svm)

                best_avg_mse_svm = avg_mse
                best_params_svm_mse = param_dict
                y_pred_best_svm_mse = y_pred_test_svm

            print(f"SVM Params: {param_dict}, Accuracy: {avg_acc}, MSE: {avg_mse}")




    # Definizione degli iperparametri per XGBoost
    grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # Testiamo tutte le combinazioni di iperparametri per XGBoost
    best_avg_acc_xgb=0
    xgb_tr_acc_ba = 0
    xgb_ts_acc_ba = 0
    xgb_tr_mse_ba = 0
    xgb_ts_mse_ba = 0
    best_params_xgb_acc={}
    y_pred_best_xgb_acc=[]

    best_avg_mse_xgb=inf
    xgb_tr_acc_br = 0
    xgb_ts_acc_br = 0
    xgb_tr_mse_br = 0
    xgb_ts_mse_br = 0
    best_params_xgb_mse={}
    y_pred_best_xgb_mse=[]

    best_eval_results_acc = {}
    best_eval_results_mse = {}

    for params in itertools.product(*grid_xgb.values()):
        
        avg_acc=0
        avg_mse=0
        
        param_dict = dict(zip(grid_xgb.keys(), params))

        acc_scores = []
        mse_scores = []

        for train_index, val_index in skf.split(X_train_encoded_full, y_train_full):
                X_train, X_val = X_train_encoded_full[train_index], X_train_encoded_full[val_index]
                y_train, y_val = y_train_full[train_index], y_train_full[val_index]

                xgb = XGBClassifier(**param_dict, eval_metric=['logloss', 'rmse', 'error'])
                xgb.fit(X_train, y_train, verbose=False)
                y_pred_xgb = xgb.predict(X_val)

                acc_scores.append(accuracy_score(y_val, y_pred_xgb))
                mse_scores.append(mean_squared_error(y_val, y_pred_xgb))

        
        avg_acc = np.mean(acc_scores)
        if(avg_acc>best_avg_acc_xgb):
            
            #refit
            eval_results = {}
            xgb = XGBClassifier(**param_dict, eval_metric=['logloss', 'rmse', 'error'])
            xgb.fit(X_train_encoded_full, y_train_full, eval_set=[(X_train_encoded_full, y_train_full),(X_test_encoded, y_test)], verbose=False)
            eval_results = xgb.evals_result()

            y_pred_test_xgb = xgb.predict(X_test_encoded)
            y_pred_train_xgb = xgb.predict(X_train_encoded_full)

            xgb_tr_acc_ba = accuracy_score(y_train_full, y_pred_train_xgb)
            xgb_ts_acc_ba = accuracy_score(y_test, y_pred_test_xgb)
            xgb_tr_mse_ba = mean_squared_error(y_train_full, y_pred_train_xgb)
            xgb_ts_mse_ba = mean_squared_error(y_test, y_pred_test_xgb)

            best_avg_acc_xgb = avg_acc
            best_params_xgb_acc = param_dict
            y_pred_best_xgb_acc = y_pred_test_xgb
            best_eval_results_acc = eval_results

        avg_mse = np.mean(mse_scores)
        if(avg_mse<best_avg_mse_xgb):

            #refit
            eval_results = {}
            xgb = XGBClassifier(**param_dict, eval_metric=['logloss', 'rmse', 'error'])
            xgb.fit(X_train_encoded_full, y_train_full, eval_set=[(X_train_encoded_full, y_train_full),(X_test_encoded, y_test)], verbose=False)
            eval_results = xgb.evals_result()

            y_pred_test_xgb = xgb.predict(X_test_encoded)
            y_pred_train_xgb = xgb.predict(X_train_encoded_full)

            xgb_tr_acc_br = accuracy_score(y_train_full, y_pred_train_xgb)
            xgb_ts_acc_br = accuracy_score(y_test, y_pred_test_xgb)
            xgb_tr_mse_br = mean_squared_error(y_train_full, y_pred_train_xgb)
            xgb_ts_mse_br = mean_squared_error(y_test, y_pred_test_xgb)

            best_avg_mse_xgb = avg_mse
            best_params_xgb_mse = param_dict
            y_pred_best_xgb_mse = y_pred_test_xgb
            best_eval_results_mse = eval_results

        print(f"XGBoost Params: {param_dict}, Accuracy: {avg_acc}, MSE: {avg_mse}")


    print("SVM Best Params for accurancy:", best_params_svm_acc)
    print("SVM Accuracy VAL:", best_avg_acc_svm)
    print(f"SVM Best Accurancy Performance:  TR ACC={svm_tr_acc_ba}, TS ACC={svm_ts_acc_ba}, TR MSE={svm_tr_mse_ba}, TS MSE={svm_ts_mse_ba}")
    print("SVM Report:\n", classification_report(y_test, y_pred_best_svm_acc))

    print("SVM Best Params for MSE:", best_params_svm_mse)
    print("SVM MSE VAL:", best_avg_mse_svm)
    print(f"SVM Best MSE Performance:  TR ACC={svm_tr_acc_br}, TS ACC={svm_ts_acc_br}, TR MSE={svm_tr_mse_br}, TS MSE={svm_ts_mse_br}")
    print("SVM Report:\n", classification_report(y_test, y_pred_best_svm_mse))

    print("XGB Best Params for accurancy:", best_params_xgb_acc)
    print("XGB Accuracy VAL:", best_avg_acc_xgb)
    print(f"XGB Best Accurancy Performance:  TR ACC={xgb_tr_acc_ba}, TS ACC={xgb_ts_acc_ba}, TR MSE={xgb_tr_mse_ba}, TS MSE={xgb_ts_mse_ba}")
    print("XGB Report:\n", classification_report(y_test, y_pred_best_xgb_acc))

    print("XGB Best Params for MSE:", best_params_xgb_mse)
    print("XGB MSE VAL:", best_avg_mse_xgb)
    print(f"XGB Best MSE Performance: TR ACC={xgb_tr_acc_br}, TS ACC={xgb_ts_acc_br}, TR MSE={xgb_tr_mse_br}, TS MSE={xgb_ts_mse_br}")
    print("XGB Report:\n", classification_report(y_test, y_pred_best_xgb_mse))

    return best_eval_results_acc, best_eval_results_mse

