from cgi import test
import random
import time
import pickle
import os
from collections import defaultdict, namedtuple
import copy

import pandas as pd
import cv2
# import optuna  # pip install optuna
# from optuna.integration import LightGBMPruningCallback
# from sklearn.metrics import log_loss
# from sklearn.model_selection import StratifiedKFold
# from sklearn import svm
# from sklearn.metrics import classification_report
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from joblib import dump, load
import lightgbm as lgb
import numpy as np
import yaml

from test_environment import TestEnvironment
import inference
import region_detection as roi
import metrics_obtention as mo
# import train as t

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
        # video = cv2.VideoCapture("images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/001/noglasses/nonsleepyCombination.avi")
        # valid, image = video.read()
        image = cv2.imread("images/test2.jpg")
        frame_metrics, image, _ = mo.process_frame(image, config["metric_obtention"])
        print(frame_metrics)
        cv2.imshow("", image)
        cv2.waitKey()

    if False:
        results = test_environment.test_open_close_eye_detection()
        #print(results["predictions"])
        #print(results["performance_metrics"])
        #results = test_environment.test_yawn_detection(naive_model)
        print(results["predictions"])
        print(results["performance_metrics"])

    if False:
        lgb_model = load("lgb_models/lgb_model_0.joblib")
        inference.inference_on_webcam(lgb_model, features, config["metric_obtention"])
    
    if False:
        dlib = True
        lgb_model = load("lgb_models/lgb_model_0.joblib")
        video_folder = "images/UTA_dataset/"
        video_paths = [
            #"images/DROZY_dataset/videos_i8/6-1.mp4",
            #"images/UTA_dataset/Fold2_part1/13/0.mp4",
            #"images/UTA_dataset/Fold2_part1/17/10.mp4",
            #"images/UTA_dataset/Fold2_part1/15/10.mp4",
            #"images/UTA_dataset/Fold2_part1/17/0.mp4",
            # "images/UTA_dataset/Fold5_part1/52/0.mov",
            #"images/UTA_dataset/Fold5_part1/53/0.MOV",
            #"images/UTA_dataset/Fold5_part1/49/0.mp4",
            # "images/UTA_dataset/Fold3_part2/33/10.mp4",
            #"images/UTA_dataset/Fold3_part1/28/10.MOV",
            # "images/adrian/adrian3.mp4",
            "images/adrian/facha.mp4",
            # "images/adrian/new_blink.mp4",
            # "images/adrian/ducha_fria.mp4",
            # "images/adrian/yarify.mp4",
            # "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/036/noglasses/nonsleepyCombination.avi",
            # "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/034/noglasses/yawning.avi",
            # "images/NTHUDDD_dataset/Testing_Dataset/016/noglasses/016_noglasses_mix.mp4",
            # "images/eyeblink8/1/26122013_223310_cam.avi",
            # "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Evaluation Dataset/022/022_noglasses_mix.mp4",
        ]
        video_names = [ f"{dlib}{path.split('/')[-3]}_{path.split('/')[-2]}_{path.split('/')[-1][:-3]}avi" for path in video_paths ]
        videos = [ cv2.VideoCapture(video_path) for video_path in video_paths ]
        test_environment.test_open_close_eye_detection_videos(lgb_model, features, videos, config["metric_obtention"], video_names=video_names)

    if False:
        df_path = "NTHUDDD_dataset/train/033_glasses_sleepy.csv"
        df = pd.read_csv(df_path)
        result_df = mo.obtain_metrics_from_df(df, config["metric_obtention"])
        result_df.to_csv("cosa.csv")

    if False:
        videos_and_labels = test_environment.get_videos_and_labels_NTHUDDD()
        df_dataset_dict = mo.create_dataset_from_videos_NTHU(videos_and_labels, target_folder="NTHUDDD_dataset/", config=config["metric_obtention"])

    if False:
        df_dataset_dict = mo.create_dataset_from_df_NTHU(source_folder="NTHUDDD_dataset/", target_folder="NTHUDDD_dataset_step2/", config=config["metric_obtention"])

    ### EXPERIMENTO 1: PERFORMANCE DETECCION ESTADO DEL OJO (ABIERTO/CERRADO) DE LOS 3 METODOS PROBADOS
    if False:
        # images, labels, _ = test_environment.prepare_CEW_dataset()
        # source_df = mo.create_dataset_from_images_CEW(images, labels, target_folder="CEW_dataset/", config=config["metric_obtention"])
        source_df = pd.read_csv("CEW_dataset/big_df.csv")
        df_list = []

        method_ind_list = [1, 2, 3]
        method_threshold_list = [0.2, 0.2, 0.2]
        method_gray_zone_list = [0, 0, 0]
        for threshold_val in range(0, 80, 5):
            threshold = threshold_val / 100
            print(f"THRESHOLD: {threshold}")
            for ind, method in enumerate(method_ind_list):
                config_copy = copy.deepcopy(config)
                config_copy["metric_obtention"]["eye_closure_method"] = method
                config_copy["metric_obtention"]["eye_closure_threshold"] = threshold
                config_copy["metric_obtention"]["gray_zone_perc"] = method_gray_zone_list[ind]
                performance_metrics = test_environment.test_open_close_eye_detection(source_df=source_df, config=config_copy["metric_obtention"])
                print(f"METHOD {method}:")
                print(performance_metrics)      
                df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                df["method"] = method
                df["threshold"] = threshold
                df.to_csv("test2.csv")
                df_list.append(df)
        
        big_df = pd.concat(df_list)
        big_df.to_csv("experiment1.csv")
    
    ### EXPERIMENTOS 2 Y 3: PERFORMANCE DETECCION PARPADEOS SOBRE VIDEOS DE LOS 3 METODOS PROBADOS
    if False:
        # videos_and_labels = test_environment.get_videos_and_labels_eyeblink8()
        # big_df = mo.create_dataset_from_videos_eyeblink8(videos_and_labels=videos_and_labels, target_folder="eyeblink8_dataset/", config=config["metric_obtention"])
        # big_df.to_csv("eyeblink8_dataset/big_df.csv")
        experiment = 2
        
        source_df = pd.read_csv("eyeblink8_dataset/big_df.csv")
        method_ind_list = [1, 2, 3]
        method_threshold_list = [0.45, 0.45, 0.45]
        method_gray_zone_list = [0.1, 0.1, 0.1]
        df_list = []

        for threshold_perc_val in range(40, 76, 4):
            for gray_zone_val in range(0, 225, 25):
                gray_zone_perc = gray_zone_val / 1000
                main_threshold_perc = threshold_perc_val / 100
                print(f"THRESHOLD_PERC: {main_threshold_perc}")
                print(f"GRAYZONE_PERC: {gray_zone_perc}")
                for ind, method in enumerate(method_ind_list):
                    config_copy = copy.deepcopy(config)
                    config_copy["metric_obtention"]["eye_closure_method"] = method
                    config_copy["metric_obtention"]["main_threshold_perc"] = main_threshold_perc
                    config_copy["metric_obtention"]["gray_zone_perc"] = gray_zone_perc

                    if experiment == 2:
                        performance_metrics = test_environment.test_blink_detection_eyeblink8(source_df, config_copy["metric_obtention"])
                    elif experiment == 3:
                        performance_metrics = test_environment.test_open_close_eye_detection_eyeblink8(source_df, config_copy["metric_obtention"])

                    print(f"METHOD {method}:")
                    print(performance_metrics)
                    df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                    df["method"] = method
                    df["threshold_perc"] = main_threshold_perc
                    df["gray_zone_perc"] = gray_zone_perc
                    df.to_csv(f"test{experiment}.csv")
                    df_list.append(df)
        
        big_df = pd.concat(df_list)
        big_df.to_csv(f"experiment{experiment}.csv")

    ### EXPERIMENTOS 5: PERFORMANCE DETECCION BOSTEZOS SOBRE VIDEOS DEL UNICO METODO ACTUAL
    if False:
        # videos_and_labels = test_environment.get_videos_and_labels_eyeblink8()
        # big_df = mo.create_dataset_from_videos_eyeblink8(videos_and_labels=videos_and_labels, target_folder="eyeblink8_dataset/", config=config["metric_obtention"])
        # big_df.to_csv("eyeblink8_dataset/big_df.csv")
        experiment = 5
        source_df = pd.read_csv("NTHUDDD_dataset/big_df.csv")
        method_ind_list = [1, 2]
        # method_threshold_list = [0.45, 0.45, 0.45]
        # method_gray_zone_list = [0.1, 0.1, 0.1]
        df_list = []
        for threshold_val in range(36, 68, 4):
            for gray_zone_val in range(0, 250, 25):
                for num_past_frames_mouth in range(20, 70, 10):
                    main_threshold = threshold_val / 100
                    gray_zone_mouth = gray_zone_val / 1000
                    print(f"THRESHOLD_PERC: {main_threshold}")
                    for ind, method in enumerate(method_ind_list):
                        config_copy = copy.deepcopy(config)
                        config_copy["metric_obtention"]["mouth_closure_method"] = method
                        config_copy["metric_obtention"]["mouth_yawn_threshold"] = main_threshold
                        config_copy["metric_obtention"]["gray_zone_mouth"] = gray_zone_mouth
                        config_copy["metric_obtention"]["num_past_frames_mouth"] = num_past_frames_mouth

                        performance_metrics = test_environment.test_yawn_detection_NTHUDDD(source_df, config_copy["metric_obtention"])
                        
                        print(f"METHOD {method}:")
                        print(performance_metrics)
                        df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                        df["method"] = method
                        df["threshold_perc"] = main_threshold
                        df["gray_zone"] = gray_zone_mouth
                        df["num_past_frames"] = num_past_frames_mouth
                        df.to_csv(f"test{experiment}.csv")
                        df_list.append(df)
        
        big_df = pd.concat(df_list)
        big_df.to_csv(f"experiment{experiment}.csv")
    
    ### EXPERIMENTOS 6: PERFORMANCE DETECCION CABECEOS SOBRE VIDEOS
    if True:
        # videos_and_labels = test_environment.get_videos_and_labels_eyeblink8()
        # big_df = mo.create_dataset_from_videos_eyeblink8(videos_and_labels=videos_and_labels, target_folder="eyeblink8_dataset/", config=config["metric_obtention"])
        # big_df.to_csv("eyeblink8_dataset/big_df.csv")
        experiment = 6
        source_df = pd.read_csv("NTHUDDD_dataset/big_df.csv")
        method_ind_list = [1]
        df_list = []
        for threshold_perc_val in range(60, 90, 4):
            # for gray_zone_val in range(20, 70, 50):
                # for num_past_frames in range(20, 30, 10):
                    main_threshold_perc = threshold_perc_val / 100
                    # gray_zone = gray_zone_val / 1000
                    print(f"THRESHOLD_PERC: {main_threshold_perc}")
                    for ind, method in enumerate(method_ind_list):
                        config_copy = copy.deepcopy(config)
                        # config_copy["metric_obtention"]["pitch_method"] = method
                        config_copy["metric_obtention"]["head_nod_threshold_perc"] = main_threshold_perc
                        # config_copy["metric_obtention"]["gray_zone_pitch"] = gray_zone
                        # config_copy["metric_obtention"]["num_past_frames_pitch"] = num_past_frames

                        performance_metrics = test_environment.test_headnod_detection_NTHUDDD(source_df, config_copy["metric_obtention"])
                        
                        print(f"METHOD {method}:")
                        print(performance_metrics["total"])
                        df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                        df["method"] = method
                        df["threshold_perc"] = main_threshold_perc
                        # df["gray_zone"] = gray_zone
                        # df["num_past_frames"] = num_past_frames
                        # df["gray_zone"] = gray_zone_mouth
                        # df["num_past_frames"] = num_past_frames_mouth
                        df.to_csv(f"test{experiment}.csv")
                        df_list.append(df)
        
        big_df = pd.concat(df_list)
        big_df.to_csv(f"experiment{experiment}.csv")

    if False:

        video_path = "images/DROZY_dataset/videos_i8/1-1.mp4"
        df = mo.create_dataset_from_video(cv2.VideoCapture(video_path), 4, config=config["metric_obtention"], label=1)
        df.to_csv("testDROZY.csv")
        # name_list = ["49_0.csv", "49_10_1.csv", "49_10_2.csv", "51_0.csv", "51_10.csv", "52_0.csv", "52_10.csv", "53_0.csv", "53_10.csv", "54_0.csv", "54_10.csv"]
        #big_df.to_csv(f"UTA_dataset_pupil/big_df")
        # for ind, df in enumerate(df_list):
        #     df.to_csv(f"UTA_dataset/{name_list[ind]}")

#TODO: a la hora de determinar los tests, podemos usar metricas de falsos positivos, 
# falsos negativos y tener preferencia por los falsos positivos --> es mejor detectar un drowsy cuando no es cierto que al reves
