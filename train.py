import random
import time
import pickle
import os
from collections import defaultdict

import pandas as pd
import cv2
import optuna  # pip install optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
#import eulerian_magnification as em
#from eulerian_magnification.io import load_video_float
from joblib import dump, load
import lightgbm as lgb
import numpy as np

from test_environment import TestEnvironment
from models import naive_model, knn_model, lgb_model
import region_detection as roi

def train_knn_classifier(df, features, save=False):
    data = df[features]
    labels = df["label"]
    labels = labels.replace(to_replace=10, value=1)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=30)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    print(classification_report(y_test, y_pred))

    if save:
        dump(knn, "knn_model.joblib")
    return knn


def train_lgb_classifier(df, features, save=False):
    data = df[features]
    labels = df["label"]
    labels = labels.replace(to_replace=10, value=1)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    lgb_model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        num_leaves=4000,
        num_iterations=1000,
        learning_rate=0.01,
        verbosity=1,
    )
    lgb_model.fit(x_train, y_train)

    y_pred = lgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))

    if save:
        dump(lgb_model, "lgb_model.joblib")
    return lgb_model


def train_lgb_optuna(df, features, save=False):
    def objective(trial, X, y):
        param_grid = {
            "device_type": trial.suggest_categorical("device_type", ['gpu']),
            "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
            "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.2, 0.95, step=0.1
            ),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.2, 0.95, step=0.1
            ),
        }

        num_splits = 4
        cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=1121218)

        cv_scores = np.empty(num_splits)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = lgb.LGBMClassifier(objective="binary", **param_grid)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="binary_logloss",
                early_stopping_rounds=100,
                # callbacks=[
                #     LightGBMPruningCallback(trial, "binary_logloss")
                # ],  # Add a pruning callback
            )
            preds = model.predict_proba(X_test)
            cv_scores[idx] = log_loss(y_test, preds)

        return np.mean(cv_scores)

    data = df[features]
    labels = df["label"]
    labels = labels.replace(to_replace=10, value=1)

    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, data, labels)
    study.optimize(func, n_trials=20)

    best_params = {
        "objective": "binary",
        "metric": "accuracy",
        "verbosity": -1,
        "boosting_type": "dart",
        "seed": 42
    } 
    best_params.update(study.best_params)
    print(f"\tBest value (rmse): {study.best_value:.5f}")
    print(f"\tBest params:")

    for key, value in best_params.items():
        print(f"\t\t{key}: {value}")
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    train_dataset = lgb.Dataset(
        data=x_train,
        label=y_train,
    )
    lgb_model = lgb.train(best_params,
                          train_dataset,
                          num_boost_round=4000,
                          ) 

    y_pred = lgb_model.predict(x_test)
    print(classification_report(y_test, y_pred))   

    if save:
        dump(lgb_model, "lgb_model_optuna.joblib")
    return lgb_model