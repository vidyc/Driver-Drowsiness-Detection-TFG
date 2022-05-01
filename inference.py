import math
import random
import time
import os
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np

import cv2
import pandas as pd

import region_detection as roi
import image_analysis
import metrics_obtention as mo


def inference_on_dataset(df, model, model_features):
    x_data = df[model_features]
    return [round(p) for p in model.predict(x_data)]


def inference_on_video(input_video, model, model_features, path="output_video.avi", max_num_frames=float('inf'), period_length=1):  
    max_width = 1080
    frame_metrics = {}
    global_metrics = {}
    periodical_data = { 
                        "frame_count" : 0,
                        "closed_eye_frame_count" : 0,
                        "current_frames_closed_eyes" : 0,
                        "max_frames_closed_eyes" : 0,
                        "mean_frames_closed_eyes" : 0,
                        "num_blinks" : 0,
                        "num_yawns" : 0,
                        "previous_frame_eye_state" : None,
                        #"ear_values" : [], 
                        "sum_ear": 0,
                        "sum_left_iris_diameter": 0,
                        "sum_right_iris_diameter": 0,
                       }

    debug = False
    predictions = []
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print(fps)
    frames_per_minute = int(fps * 60)
    remaining_frames_of_period = period_length

    width  = int(input_video.get(3))   # float `width`
    height = int(input_video.get(4))  # float `height`
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))  

    font_scale = min(1, width / max_width)

    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame and periodical_data["frame_count"] < max_num_frames:
        remaining_frames_of_period -= 1
        obtain_global_metrics = False
        if remaining_frames_of_period <= 0:
            obtain_global_metrics = True
            remaining_frames_of_period = period_length

        current_metrics = mo.obtain_frame_metrics(frame, periodical_data, obtain_global_metrics, fps)
        frame_metrics = current_metrics["frame_metrics"]
        periodical_data = current_metrics["periodical_data"]
        
        # TODO: pasar un parametro o de alguna manera decirle a la funcion que metricas obtener
        if obtain_global_metrics:
            if frame_metrics is not None:
                global_metrics = current_metrics["global_metrics"]
                selected_metrics = global_metrics
                x_data = np.array(list({ metric:value for metric, value in selected_metrics.items() if metric in model_features }.values())).reshape(1, -1)
                drowsiness_state = model.predict(x_data)
                predictions.append(drowsiness_state)
            else:
                drowsiness_state = -1
                predictions.append(drowsiness_state)

        info_to_show = {
            "frame_count": periodical_data["frame_count"],
            "closed_eye_frame_count": periodical_data["closed_eye_frame_count"],
            "perclos": global_metrics["perclos"],
            "current_frames_closed_eyes": periodical_data["current_frames_closed_eyes"],
            "num_blinks": periodical_data["num_blinks"],
            "blinks_per_minute": global_metrics["blinks_per_minute"],
            "previous_eye_state": periodical_data["previous_frame_eye_state"],
            "current_eye_state": periodical_data["current_eye_state"],
            "right_EAR": frame_metrics["rear"],
            "left_EAR": frame_metrics["lear"],
            "num_yawns": periodical_data["num_yawns"],
            "yawns_per_minute": global_metrics["yawns_per_minute"],
            "prediction": drowsiness_state,
        }

        edited_frame = frame.copy()
        point = ( int(0), int(0.05 * height) )
        for metric, value in info_to_show.items():
            edited_frame = cv2.putText(edited_frame, f"{metric}: {value}", point, color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale)
            point = (point[0], point[1] + int(0.05*height))

        out.write(edited_frame)
        valid_frame, frame = input_video.read()
        
        if periodical_data["frame_count"] % 1000 == 0:
            print(f"{periodical_data['frame_count']}: {time.time() - start}")
            print(periodical_data)
            print(global_metrics)
            print(drowsiness_state)

    out.release()
    print(frame_metrics)
    print()
    print(periodical_data)
    print()
    print(global_metrics)

    return predictions