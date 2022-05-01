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
import yaml

from test_environment import TestEnvironment
from models import naive_model, knn_model, lgb_model
import region_detection as roi
import metrics_obtention as mo
import train as t

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
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.loader.FullLoader)

    features = config["train_inference"]["model_features"]    

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

    if False:

        path = "UTA_dataset_pupil/"
        # 3 folds para train y 2 para validation
        train_subjects = range(13, 31, 1)
        train_dfs = []
        for subject in train_subjects:
            train_dfs.append(pd.read_csv(f"{path}{subject}_0.csv"))
            train_dfs.append(pd.read_csv(f"{path}{subject}_10.csv"))
        
        test_subjects = range(13, 37, 1)
        test_dfs = []
        for subject in train_subjects:
            test_dfs.append(pd.read_csv(f"{path}{subject}_0.csv"))
            test_dfs.append(pd.read_csv(f"{path}{subject}_10.csv"))
            if subject == 32 or subject == 49:
                test_dfs.append(pd.read_csv(f"{path}{subject}_10_2.csv"))

        train_df = pd.concat(train_dfs)
        x_train = train_df[features]
        y_train = train_df["label"]
        y_train = y_train.replace(to_replace=10, value=1)
        test_df = pd.concat(test_dfs)
        x_test = test_df[features]
        y_test = test_df["label"]
        y_test = y_test.replace(to_replace=10, value=1)

        lgb_model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            num_leaves=40,
            num_iterations=1000,
            learning_rate=0.01,
            verbosity=1,
            #early_stopping=20,
        )
        lgb_model.fit(x_train, y_train)
        dump(lgb_model, "lgb_model_0.joblib")
        y_pred = lgb_model.predict(x_test)
        print(classification_report(y_test, y_pred))

        num_hits = 0
        num = 0
        for ind in range(0, 1000):
            y_pred = lgb_model.predict(x_test.iloc[ind].to_numpy().reshape(1, -1))
            y_test1 = y_test.iloc[ind]
            print(y_pred)
            print(y_test1)
            if y_pred == y_test1:
                num_hits += 1
            num += 1

        print(num_hits/num)

    if False:
        df = pd.read_csv("UTA_dataset_pupil/big_df.csv")
        old_df = pd.read_csv("UTA_dataset2/big_df.csv")
        df2 = df.groupby("label").mean()
        print(df2)
        old_df2 = old_df.groupby("label").mean()
        print(old_df2)
        #data = df.drop(["label", "frame", "blink_frequency", "mean_ear"], axis = 1)
        data = df[features]
        labels = df["label"]
        labels = labels.replace(to_replace=10, value=1)
        old_data = old_df[features]
        old_labels = old_df["label"]
        old_labels = old_labels.replace(to_replace=10, value=1)
        knn = KNeighborsClassifier(n_neighbors=30)
        lgb_model = lgb.LGBMClassifier(
             boosting_type="gbdt",
             num_leaves=4000,
             num_iterations=1000,
             learning_rate=0.01,
        )
        print(cross_val_score(lgb_model, data, labels, cv=5, scoring="accuracy"))
        print(cross_val_score(lgb_model, old_data, old_labels, cv=5, scoring="accuracy"))
       # print(cross_val_score(lgb_model, data, labels, cv=5, scoring="accuracy"))

    if False:
        image = cv2.imread("images/xi-yinping.jpg")
        frame_metrics, image = mo.process_frame(image)
        print(frame_metrics)
        cv2.imshow("", image)
        cv2.waitKey()

    if False:
        #results = test_environment.test_open_close_eye_detection(naive_model)
        #print(results["predictions"])
        #print(results["performance_metrics"])
        results = test_environment.test_yawn_detection(naive_model)
        print(results["predictions"])
        print(results["performance_metrics"])


    if True:
        lgb_model = load("lgb_model_0.joblib")
        video_folder = "images/UTA_dataset/"
        video_paths = [
            "images/UTA_dataset/Fold1_part1/01/10.mov",
            #"images/UTA_dataset/Fold2_part1/13/0.mp4",
            #"images/UTA_dataset/Fold2_part1/17/10.mp4",
            #"images/UTA_dataset/Fold2_part1/15/10.mp4",
            #"images/UTA_dataset/Fold2_part1/17/0.mp4",
            #"images/UTA_dataset/Fold5_part1/52/0.mov",
            #"images/UTA_dataset/Fold5_part1/53/0.MOV",
            #"images/UTA_dataset/Fold5_part1/49/0.mp4",
            #"images/UTA_dataset/Fold3_part1/28/0.MOV",
            #"images/UTA_dataset/Fold3_part1/28/10.MOV",
            #"images/adrian/test.mp4"
        ]
        video_names = [ f"{path.split('/')[-2]}_{path.split('/')[-1][:-3]}avi" for path in video_paths ]
        videos = [ cv2.VideoCapture(video_path) for video_path in video_paths ]
        test_environment.test_open_close_eye_detection_videos(lgb_model, features, videos, num_frames=2000, video_names=video_names)


    if False:
        # add subject column to existing dataframes
        path = "UTA_dataset_pupils/"
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
        path = "UTA_dataset_pupil/"
        df_list = []
        for filename in os.listdir(path):
            if filename != "big_df.csv":
                file = os.path.join(path, filename)
                if os.path.isfile(file) and ".csv" in filename:
                    df = pd.read_csv(file)
                    df_list.append(df)
        
        df = pd.concat(df_list)
        df.to_csv(f"{path}big_df.csv", index=False)

    if False:

        video_path = "images/UTA_dataset/"
        df_list = lgb_model.create_dataset_from_videos(video_path)
        # name_list = ["49_0.csv", "49_10_1.csv", "49_10_2.csv", "51_0.csv", "51_10.csv", "52_0.csv", "52_10.csv", "53_0.csv", "53_10.csv", "54_0.csv", "54_10.csv"]
        big_df = pd.concat(df_list)
        big_df.to_csv(f"UTA_dataset_pupil/big_df")
        # for ind, df in enumerate(df_list):
        #     df.to_csv(f"UTA_dataset/{name_list[ind]}")

#TODO: a la hora de determinar los tests, podemos usar metricas de falsos positivos, 
# falsos negativos y tener preferencia por los falsos positivos --> es mejor detectar un drowsy cuando no es cierto que al reves
