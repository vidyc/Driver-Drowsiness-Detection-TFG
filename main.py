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
from models import naive_model, knn_model
import region_detection as roi

max_num_frames = 5400

def compute_accuracy(pred, actual):
    total_elements = len(pred)
    num_hits = 0
    failed_predictions = []
    for i, y_pred in enumerate(pred):
        if y_pred == actual[i]:
            num_hits += 1
        else:
            failed_predictions.append(i)

    return num_hits / total_elements, failed_predictions

if __name__ == "__main__":
    test_environment = TestEnvironment()

    if False:
        vid, fps = load_video_float("Eulerian_Motion_Magnification-main/source/subway.mp4")
        em.eulerian_magnification(vid, fps, 
        freq_min=50.0 / 60.0,
        freq_max=1.0,
        amplification=50,
        pyramid_levels=3
        )
        out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
        type(vid)


    features = ["perclos", "blink_frequency", "current_time_closed_eyes"]
    if False:
        # path = "images/UTA_dataset/"

        # df_list = naive_model.create_dataset_from_videos(path)

        # counter = 0
        # for df in df_list:
        #     df.to_csv(f"UTA_dataset/{counter}.csv")
        #     counter += 1

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

        df = pd.read_csv("big_df.csv")
        df2 = df.groupby("label").mean()
        print(df2)
        #data = df.drop(["label", "frame", "blink_frequency", "mean_ear"], axis = 1)
        data = df[features]
        labels = df["label"]
        labels = labels.replace(to_replace=10, value=1)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

        knn = KNeighborsClassifier(n_neighbors=30)
        knn.fit(x_train, y_train)
        # lgb = lgb.LGBMClassifier(
        #     boosting_type="dart",
        #     num_leaves=4000,
        #     num_iterations=1000,
        #     learning_rate=0.01,
        # )
        start = time.time()
        # lgb.fit(x_train, y_train)
        #svm_classifier = svm.LinearSVC(max_iter=1000)

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

        #svm_classifier.fit(x_train, y_train)
        print(f"training took {time.time() - start} seconds")
        
        train_dataset = lgb.Dataset(
            data=data,
            label=labels
        )
        lgb_model = lgb.train(best_params,
                              train_dataset,
                              num_boost_round=4000,
                              ) 

        dump(lgb_model, "lgb_model.joblib")
        dump(knn, "knn_model.joblib")
        #dump(svm_classifier, "svm_model.joblib")
        y_pred = lgb_model.predict(x_test)
        y_pred2 = knn.predict(x_test)

        print(classification_report(y_test, y_pred))
        print(classification_report(y_test, y_pred2))

        # accuracy, failed_predictions = compute_accuracy(y_pred, y_test.to_list())
        # print(accuracy)
        #print(accuracy)
        #print([ i for i in failed_predictions ])
        #print([ df.iloc[i, :] for i in failed_predictions ])
        print("n")
        # print(cross_val_score(knn, x_train, y_train, cv=4, scoring="accuracy"))
        # print(cross_val_score(knn, x_train, y_train, cv=4, scoring="f1"))
        # print(cross_val_score(knn, x_train, y_train, cv=4, scoring="recall"))
        # print(cross_val_score(knn, x_train, y_train, cv=4, scoring="precision"))
        #print(cross_validate(knn, x_train, y_train, cv=4, scoring=["accuracy", "f1", "recall", "precision"]))

        #print(knn.score(x_test, y_test))
        #print(lgb.score(x_test, y_test))
        #print(svm_classifier.score(x_test, y_test))

    if False:
        image = cv2.imread("images/test2.jpg")
        naive_model.process_frame(image)
        cv2.imshow("", image)
        cv2.waitKey()

    if False:
        #results = test_environment.test_open_close_eye_detection(naive_model)
        #print(results["predictions"])
        #print(results["performance_metrics"])
        results = test_environment.test_yawn_detection(naive_model)
        print(results["predictions"])
        print(results["performance_metrics"])


    if False:
        video = cv2.VideoCapture("images/UTA_dataset/Fold5_part1/49/0.mp4")
        naive_model.inference_on_video(video)


    if False:
        # add subject column to existing dataframes
        path = "UTA_dataset/"
        subject_dict = {
            # FOLD 1
            "0.csv": 13,
            "1.csv": 13,
            "2.csv": 14,
            "3.csv": 14,
            "4.csv": 15,
            "5.csv": 15,
            "6.csv": 16,
            "7.csv": 16,
            "8.csv": 17,
            "9.csv": 17,
            "10.csv": 18,
            "11.csv": 18,
            # FOLD 2
            "12.csv": 19,
            "13.csv": 19,
            "14.csv": 20,
            "15.csv": 20,
            "16.csv": 21,
            "17.csv": 21,
            "18.csv": 22,
            "19.csv": 22,
            "20.csv": 23,
            "21.csv": 23,
            "22.csv": 24,
            "23.csv": 24,
            # FOLD 3
            "24.csv": 25,
            "25.csv": 25,
            "26.csv": 26,
            "27.csv": 26,
            "28.csv": 27,
            "29.csv": 27,
            "30.csv": 28,
            "31.csv": 28,
            "32.csv": 29,
            "33.csv": 29,
            "34.csv": 30,
            "35.csv": 30,
            # FOLD 4
            "36.csv": 31,
            "37.csv": 31,
            "38.csv": 32,
            "39.csv": 32,
            "40.csv": 32,
            "41.csv": 33,
            "42.csv": 33,
            "43.csv": 34,
            "44.csv": 34,
            "45.csv": 35,
            "46.csv": 35,
            "47.csv": 36,
            "48.csv": 36,
            # FOLD 5
            "49_0.csv": 49,
            "49_10_1.csv": 49,
            "49_10_2.csv": 49, 
            "50_0.csv": 50,
            "50_10.csv": 50,
            "51_0.csv": 51, 
            "51_10.csv": 51,
            "52_0.csv": 52,
            "52_10.csv": 52, 
            "53_0.csv": 53,
            "53_10.csv": 53,
            "54_0.csv": 54, 
            "54_10.csv": 54,         
        }
        df_list = []
        suffix_list = ["0", "10", "10_2"]
        sub_ind = 0
        count_dict = defaultdict(lambda: -1)
        for filename in os.listdir(path):
            file = os.path.join(path, filename)
            if os.path.isfile(file) and ".csv" in filename:
                df = pd.read_csv(file)
                subject = subject_dict[filename]
                count_dict[subject] += 1
                df["subject"] = subject_dict[filename]
                df_list.append(df)
                df_name = f"UTA_dataset2/{subject_dict[filename]}_{suffix_list[count_dict[subject]]}.csv"
                df.to_csv(df_name)

    if False:
        path = "UTA_dataset2/"
        df_list = []
        for filename in os.listdir(path):
            file = os.path.join(path, filename)
            if os.path.isfile(file) and ".csv" in filename:
                df = pd.read_csv(file)
                df_list.append(df)
        
        df = pd.concat(df_list)
        df.to_csv("UTA_dataset2/big_df.csv", index=False)

    if True:

        # video_path = "images/UTA_dataset/Fold5_part1/"
        # df_list = knn_model.create_dataset_from_videos(video_path)
        # name_list = ["49_0.csv", "49_10_1.csv", "49_10_2.csv", "51_0.csv", "51_10.csv", "52_0.csv", "52_10.csv", "53_0.csv", "53_10.csv", "54_0.csv", "54_10.csv"]

        # for ind, df in enumerate(df_list):
        #     df.to_csv(f"UTA_dataset/{name_list[ind]}")

        knn = load("knn_model.joblib")
        lgb = load("lgb_model.joblib")
        knn_scores = 0
        lgb_scores = 0
        naive_scores = 0
        path = "UTA_dataset/"
        df_list = []
        study_cases = {}
        for filename in os.listdir(path):
            file = os.path.join(path, filename)
            if os.path.isfile(file) and ".csv" in filename:
                df = pd.read_csv(file)
                print(file)
                x_data = df[features]
                y_data = df["label"].replace(to_replace=10, value=1)
                #naive_score = naive_model.score(naive_x_data, y_data)
                knn_pred = knn.predict(x_data)
                lgb_pred = lgb.predict(x_data)
                lgb_bin_preds = [ round(pred) for pred in lgb_pred ]
                #print(lgb_bin_preds)
                knn_score = knn.score(x_data, y_data)
                lgb_score, _ = compute_accuracy(lgb_bin_preds, y_data)
                #print(naive_score)
                print(knn_score)
                print(lgb_score)
                if lgb_score < 0.6:
                    study_cases[file] = lgb_score
                df_list.append(df)
        
        print(study_cases)

        df = pd.concat(df_list)
        x_data = df[features].to_numpy()
        y_data = df["label"].replace(to_replace=10, value=1)
        
        #naive_x_data = df[["perclos", "blinks_per_minute", "current_time_closed_eyes"]].to_numpy()

        #naive_score = naive_model.score(naive_x_data, y_data)
        knn_pred = knn.predict(x_data)
        lgb_pred = lgb.predict(x_data)
        lgb_bin_preds = [ round(pred) for pred in lgb_pred ]
        #print(lgb_bin_preds)
        knn_score = classification_report(y_data, knn_pred)
        lgb_score = classification_report(y_data, lgb_bin_preds)
        #print(naive_score)
        print(knn_score)
        print(lgb_score)

        # print("ANSWERS")
        # print(results["predictions"])
        # print(dummy_labels)
        # print(results["performance_metrics"])

#TODO: a la hora de determinar los tests, podemos usar metricas de falsos positivos, 
# falsos negativos y tener preferencia por los falsos positivos --> es mejor detectar un drowsy cuando no es cierto que al reves
