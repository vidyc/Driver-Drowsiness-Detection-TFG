from cgi import test
import math
import random
import time
import pickle
import os
from collections import defaultdict, namedtuple
import copy
import optuna

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

    def compute_metrics_from_intervals(real_intervals, method_intervals):
        num_real_intervals = len(real_intervals)
        num_method_intervals = len(method_intervals)

        precision = None
        recall = None
        f1_score = None
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        ind_real = 0
        ind_method = 0
        while ind_real < num_real_intervals and ind_method < num_method_intervals:
            real_interval = real_intervals[ind_real]
            method_interval = method_intervals[ind_method]

            if set(real_interval).intersection(set(method_interval)):
                true_positives += 1
                ind_real += 1
                ind_method += 1
            elif method_interval[-1] > real_interval[-1]:
                false_negatives += 1
                ind_real += 1
            else:
                false_positives += 1
                ind_method += 1
            
        for i in range(ind_real, num_real_intervals):
            false_negatives += 1

        for i in range(ind_method, num_method_intervals):
            false_positives += 1
        
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)

        if recall is not None and precision is not None and (recall + precision) > 0:
            f1_score = 2 * (recall * precision) / (recall + precision)
        
        return {"recall": recall, "precision": precision, "f1_score": f1_score}

    if False:
        csv_path = "NTHUDDD_dataset2/train/008_noglasses_slowBlinkWithNodding.csv"
        df = pd.read_csv(csv_path)
        processed_df = mo.obtain_metrics_from_df(df, config["metric_obtention"])

        target_col = "head"
        method_col = "head_nod"

        yawn_states = list(processed_df[method_col])
        real_yawn_states = list(processed_df[target_col])
        method_yawn_intervals = []
        yawn_set = []
        previous_yawn_state = None
        for frame_number, yawn_state in enumerate(yawn_states):
            if math.isnan(yawn_state):
                continue

            if yawn_state:
                yawn_set.append(frame_number)
            
            if  not yawn_state and previous_yawn_state:
                method_yawn_intervals.append(yawn_set)
                yawn_set = []

            previous_yawn_state = yawn_state
        
        yawn_intervals = []
        yawn_set = []
        previous_yawn_state = []
        for frame_number, yawn_state in enumerate(real_yawn_states):
            if math.isnan(yawn_state):
                continue

            if yawn_state == 1:
                yawn_set.append(frame_number)
            
            if yawn_state != 1 and previous_yawn_state == 1:
                yawn_intervals.append(yawn_set)
                yawn_set = []

            previous_yawn_state = yawn_state

        print(f"real num_{target_col}: {len(yawn_intervals)}")
        print(f"real {target_col}_intervals: {yawn_intervals}")
        print(f"method num_{target_col}: {len(method_yawn_intervals)}")
        print(f"method {target_col}_intervals: {method_yawn_intervals}")
        print(compute_metrics_from_intervals(yawn_intervals, method_yawn_intervals))


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

    if True:
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
            # "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/008/noglasses/slowBlinkWithNodding.avi",
            # "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/001/noglasses/nonsleepyCombination.avi",
            # "images/NTHUDDD_dataset/Testing_Dataset/028/noglasses/028_noglasses_mix.mp4",
            # "images/eyeblink8/11/27122013_154548_cam.avi",
            # "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Evaluation Dataset/022/022_noglasses_mix.mp4",
        ]
        video_names = [ f"{dlib}{path.split('/')[-3]}_{path.split('/')[-2]}_{path.split('/')[-1][:-3]}avi" for path in video_paths ]
        videos = [ cv2.VideoCapture(video_path) for video_path in video_paths ]
        test_environment.test_open_close_eye_detection_videos(lgb_model, features, videos, config["metric_obtention"], video_names=video_names)

    if False:
        df_path = "NTHUDDD_dataset/train/001_noglasses_nonsleepy.csv"
        df = pd.read_csv(df_path)
        result_df = mo.obtain_metrics_from_df(df, config["metric_obtention"])
        result_df.to_csv("cosa.csv")

    if False:
        videos_and_labels = test_environment.get_videos_and_labels_NTHUDDD()
        df_dataset_dict = mo.create_dataset_from_videos_NTHU(videos_and_labels, target_folder="NTHUDDD_dataset2/", config=config["metric_obtention"])

    if False:
        df_dataset_dict = mo.create_dataset_from_df_NTHU(source_folder="NTHUDDD_dataset2/", target_folder="NTHUDDD_dataset2_nuevosmetodos/", config=config["metric_obtention"])

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
        # big_df = mo.create_dataset_from_videos_eyeblink8(videos_and_labels=videos_and_labels, target_folder="eyeblink8_dataset2/", config=config["metric_obtention"])
        # big_df.to_csv("eyeblink8_dataset2/big_df.csv")
        experiment = 2
        
        source_df = pd.read_csv("eyeblink8_dataset2/big_df.csv")
        method_ind_list = [1, 2, 3]
        df_list = []

        # for static_threshold_val in range(70, 100, 5):
        for threshold_perc_val in range(40, 80, 5):
            # for num_ear_past_frames in range(200, 2000, 200):
            main_threshold_perc = threshold_perc_val / 100
            for alpha_val in range(0, 100, 5):
                for gray_zone_val in range(0, 225, 25):
            # method = 1
                    # main_threshold_perc = 0.68
                    alpha_val_perc = alpha_val / 100
                    gray_zone_perc = gray_zone_val / 1000
                    # num_ear_past_frames = 200
                    # main_threshold_perc = threshold_perc_val / 100
                    # print(f"THRESHOLD: {static_threshold_val}")
                    print(f"THRESHOLD_PERC: {main_threshold_perc}")
                    # print(f"PAST_FRAMES: {num_ear_past_frames}")
                    print(f"ALPHA_VAL: {alpha_val_perc}")
                    print(f"GRAYZONE_PERC: {gray_zone_perc}")
                    for ind, method in enumerate(method_ind_list):
                        config_copy = copy.deepcopy(config)
                        config_copy["metric_obtention"]["eye_closure_method"] = method
                        # config_copy["metric_obtention"]["eye_closure_threshold"] = static_threshold_val / 100
                        config_copy["metric_obtention"]["main_threshold_perc"] = main_threshold_perc
                        config_copy["metric_obtention"]["gray_zone_perc"] = gray_zone_perc
                        config_copy["metric_obtention"]["num_ear_past_frames"] = 0
                        config_copy["metric_obtention"]["alpha_val"] = alpha_val_perc

                        if experiment == 2:
                            performance_metrics = test_environment.test_blink_detection_eyeblink8(source_df, config_copy["metric_obtention"])
                        elif experiment == 3:
                            performance_metrics = test_environment.test_open_close_eye_detection_eyeblink8(source_df, config_copy["metric_obtention"])

                        print(f"METHOD {method}:")
                        print(performance_metrics)
                        df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                        df["method"] = method
                        # df["threshold"] = static_threshold_val/100
                        df["threshold_perc"] = main_threshold_perc
                        df["gray_zone_perc"] = gray_zone_perc
                        df["alpha_val"] = alpha_val
                        df.to_csv(f"test{experiment}.csv")
                        df_list.append(df)
                
        big_df = pd.concat(df_list)
        big_df.to_csv(f"experiment{experiment}_grayzone2.csv")

    ### EXPERIMENTOS 5: PERFORMANCE DETECCION BOSTEZOS SOBRE VIDEOS DEL UNICO METODO ACTUAL
    if False:
        # videos_and_labels = test_environment.get_videos_and_labels_eyeblink8()
        # big_df = mo.create_dataset_from_videos_eyeblink8(videos_and_labels=videos_and_labels, target_folder="eyeblink8_dataset/", config=config["metric_obtention"])
        # big_df.to_csv("eyeblink8_dataset/big_df.csv")
        experiment = 5
        source_df = pd.read_csv("NTHUDDD_dataset2/big_df.csv")
        method_ind_list = [1, 2]
        # method_threshold_list = [0.45, 0.45, 0.45]
        # method_gray_zone_list = [0.1, 0.1, 0.1]
        df_list = []
        for threshold_val in range(0, 100, 5):
            for gray_zone_val in range(250, 450, 25):
                # for alpha_val in range(0, 100, 5):
                # for num_past_frames_mouth in range(0, 70, 10):
                    num_past_frames_mouth = 0
                    main_threshold = threshold_val / 100
                    gray_zone_mouth = gray_zone_val / 1000
                    # alpha_val_perc = alpha_val / 100
                    print(f"THRESHOLD_PERC: {main_threshold}")
                    for ind, method in enumerate(method_ind_list):
                        config_copy = copy.deepcopy(config)
                        config_copy["metric_obtention"]["mouth_closure_method"] = method
                        config_copy["metric_obtention"]["mouth_yawn_threshold"] = main_threshold
                        config_copy["metric_obtention"]["gray_zone_mouth"] = gray_zone_mouth
                        config_copy["metric_obtention"]["num_past_frames_mouth"] = num_past_frames_mouth
                        config_copy["metric_obtention"]["alpha_val"] = 0

                        performance_metrics = test_environment.test_yawn_detection_NTHUDDD(source_df, config_copy["metric_obtention"])
                        
                        print(f"METHOD {method}:")
                        print(performance_metrics)
                        df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                        df["method"] = method
                        df["threshold_perc"] = main_threshold
                        df["gray_zone"] = gray_zone_mouth
                        df["num_past_frames"] = num_past_frames_mouth
                        # df["alpha_val"] = alpha_val_perc
                        df.to_csv(f"test{experiment}.csv")
                        df_list.append(df)
        
        big_df = pd.concat(df_list)
        big_df.to_csv(f"experiment{experiment}_grayzone_extended.csv")
    
    ### EXPERIMENTOS 6: PERFORMANCE DETECCION CABECEOS SOBRE VIDEOS
    if False:
        # videos_and_labels = test_environment.get_videos_and_labels_eyeblink8()
        # big_df = mo.create_dataset_from_videos_eyeblink8(videos_and_labels=videos_and_labels, target_folder="eyeblink8_dataset/", config=config["metric_obtention"])
        # big_df.to_csv("eyeblink8_dataset/big_df.csv")
        experiment = 6
        source_df = pd.read_csv("NTHUDDD_dataset2/big_df.csv")

        method_ind_list = [1, 2]
        df_list = []

        def should_prune(last_scores):  
            scores = []
            for score in last_scores:
                if score is None or score < 0.5:
                    score = -1
                scores.append(score)
            
            num_scores = len(scores)
            print(scores)
            prune = True
            for i in range(1, num_scores):
                prune = prune and ( scores[i] >= scores[i - 1] )

            return False            

        previous2_f1_score = -2 ## hace 2 mediciones
        previous_f1_score = -1  ## ultima medicion
        prune = False
        for threshold_perc_val in range(75, 100, 5):
            main_threshold_perc = threshold_perc_val / 100
            num_consecutive_prunes = 0
            for gray_zone_val in range(0, 200, 25):
                gray_zone = gray_zone_val / 1000
                num_past_frames = 0

                # current_f1_score = 0
                # previous_f1_score = -1
                # previous2_f1_score = -2
                
                # if prune:
                #     num_consecutive_prunes += 1
                # else:
                #     num_consecutive_prunes = 0

                # if num_consecutive_prunes > 4:
                #     break

                for num_past_frames in range(5, 15, 5):
                    for num_past_frames_th in range(0, 2100, 600):
            # for alpha_val in range(0, 100, 5):
                # alpha_val_perc = alpha_val / 100
                        print(f"THRESHOLD_PERC: {threshold_perc_val}")
                        print(f"GRAY_ZONE: {gray_zone_val}")
                        print(f"NUM_PAST_FRAMES: {num_past_frames}")
                        print(f"NUM_PAST_FRAMES_TH: {num_past_frames_th}")
                        # print(f"ALPHA VAL: {alpha_val_perc}")
                        for ind, method in enumerate(method_ind_list):
        # method = 1
        # main_threshold_perc = 0.5
        # gray_zone = 0.05
        # num_past_frames = 10
        # num_past_frames_th = 900
                            config_copy = copy.deepcopy(config)
                            config_copy["metric_obtention"]["pitch_method"] = method
                            # config_copy["metric_obtention"]["head_nod_threshold"] = threshold_perc_val
                            config_copy["metric_obtention"]["head_nod_threshold_perc"] = main_threshold_perc
                            config_copy["metric_obtention"]["gray_zone_pitch"] = gray_zone
                            config_copy["metric_obtention"]["num_past_frames_pitch"] = num_past_frames
                            config_copy["metric_obtention"]["num_past_frames_th"] = num_past_frames_th
                            config_copy["metric_obtention"]["alpha_val"] = 0

                            performance_metrics = test_environment.test_headnod_detection_NTHUDDD(source_df, config_copy["metric_obtention"])
                            current_f1_score = performance_metrics["total"]["f1-score"]

                            print(f"METHOD {method}:")
                            print(performance_metrics["total"])
                            df = pd.DataFrame.from_dict(performance_metrics, orient="index")
                            df["method"] = method
                            df["threshold_perc"] = threshold_perc_val
                            df["gray_zone"] = gray_zone
                            df["num_past_frames"] = num_past_frames
                            df["num_past_frames_th"] = num_past_frames_th
                            # df["alpha_val"] = alpha_val_perc
                            df.to_csv(f"test{experiment}.csv")
                            df_list.append(df)
                            
                            # prune = should_prune([current_f1_score, previous_f1_score, previous2_f1_score])
                            # if prune:
                            #     print(f"PRUNING EXPLORATION... TH:{main_threshold_perc}, GZ:{gray_zone}, PF:{num_past_frames}")
                            #     print(current_f1_score)
                            #     print(previous_f1_score)
                            #     print(previous2_f1_score)
                            #     break

                            previous2_f1_score = previous_f1_score
                            previous_f1_score = current_f1_score
            
        big_df = pd.concat(df_list)
        big_df.to_csv(f"experiment{experiment}_framepast_gray+suav+dyn3.csv")

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
