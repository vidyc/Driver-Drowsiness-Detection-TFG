from email.policy import default
import math
import random
import time
import os
from collections import Counter
from collections import defaultdict
from sklearn import metrics

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


def inference_on_video(input_video, model, model_features, config, path="output_video.avi", max_num_frames=float('inf'), period_length=1):  
    max_width = 1080
    frame_metrics = {
        "rear": 0,
        "lear": 0,
        "ear": 0,
        "mean_ear_3_frames": 0,
    }
    global_metrics = defaultdict(lambda: 0)
    periodical_data = { 
                        "frame_count" : 0,
                        "closed_eye_frame_count" : 0,
                        "closed_eye_frame_values": [],
                        "current_frames_closed_eyes" : 0,
                        "previous_current_frames_closed_eyes": 0,
                        "max_frames_closed_eyes" : 0,
                        "mean_frames_closed_eyes" : 0,
                        "num_blinks" : 0,
                        "blink_values": [],
                        "previous_frame_eye_state" : None,
                        "previous2_frame_eye_state" : None,
                        "ear_values" : [], 
                        "sum_ear": 0,
                        "sum_first_ear": 0,
                        "sum_left_iris_diameter": 0,
                        "sum_right_iris_diameter": 0,
                        "current_eye_state": None,
                        "head_nod_total_frame_count": 0,
                        "previous_head_nod_state": False,
                        "head_nod_last_frame": -1000,
                        "head_nod_count": 0,
                        "head_nod_values": [],
                        "sum_first_pitch": 0,
                        "pitch_values": [],
                        "yaw_values": [],
                        "yawn_total_frame_count": 0,
                        "previous_yawn_state": False,
                        "yawn_last_frame": -1000,
                        "num_yawns" : 0,
                        "yawn_values": [],
                        "mar_values": [],
                        "nose_tip_y_values": [],
                        "mouth_top_y_values": [],
                        "sum_first_mouth_width": 0,
                        "mean_first_ear": 0,
                       }


    debug = False
    predictions = []
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print(fps)
    metrics = []
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

        current_metrics = mo.obtain_frame_metrics(frame, periodical_data, config, obtain_global_metrics, fps)
        periodical_data = current_metrics["periodical_data"]
        
        # TODO: pasar un parametro o de alguna manera decirle a la funcion que metricas obtener
        if obtain_global_metrics:
            if current_metrics["frame_metrics"] is not None:
                frame_metrics = current_metrics["frame_metrics"]
                global_metrics = current_metrics["global_metrics"]
                selected_metrics = global_metrics

                x_data = np.array(list({ metric:selected_metrics[metric] for metric in model_features }.values())).reshape(1, -1)
                # x_data = np.array(list({ metric:value for metric, value in selected_metrics.items() if metric in model_features }.values())).reshape(1, -1)
                drowsiness_state = model.predict(x_data)
                predictions.append(drowsiness_state)
                flat_metrics = dict(list(current_metrics["global_metrics"].items()) 
                                   + list(current_metrics["frame_metrics"].items()) 
                                   + list(current_metrics["periodical_data"].items()))
                selected_metrics = { metric:value for metric, value in flat_metrics.items() if metric in config["metrics_to_obtain"] }
                metrics.append(selected_metrics)
            else:
                drowsiness_state = -1
                predictions.append(drowsiness_state)
                selected_metrics = {metric:None for metric in config["metrics_to_obtain"]}
                metrics.append(selected_metrics)

        pitch_values_to_analyze = periodical_data["pitch_values"][:-config["num_frames_lag_ignore"]]
        if len(pitch_values_to_analyze) != 0:
            mean_last_pitch = round(sum(pitch_values_to_analyze)/len(pitch_values_to_analyze))
        else:
            mean_last_pitch = -1
        
        if len(periodical_data["ear_values"]) != 0:
            mean_ear_last_minute = sum(periodical_data["ear_values"])/len(periodical_data["ear_values"])
        else:
            mean_ear_last_minute = 0

        info_to_show = {
            "frame_count": periodical_data["frame_count"],
            # "closed_eye_frame_count": periodical_data["closed_eye_frame_count"],
            # "perclos": round(global_metrics["perclos"], 2),
            # "current_frames_closed_eyes": periodical_data["current_frames_closed_eyes"],
            # "num_blinks": periodical_data["num_blinks"],
            # # "blinks_per_minute": round(global_metrics["blinks_per_minute"], 2),
            # # "mean_blink_time": round(global_metrics["mean_blink_time"], 2),
            # "previous_eye_state": periodical_data["previous_frame_eye_state"],
            # "current_eye_state": periodical_data["current_eye_state"],
            # "ear": round(frame_metrics["ear"], 2),
            # "mean_ear_3_frames": round(frame_metrics["mean_ear_3_frames"], 2),
            # "mean_ear_last_minute": round(mean_ear_last_minute, 2),
            # "mean_EAR": round(periodical_data["mean_first_ear"], 2),
            "num_yawns": periodical_data["num_yawns"],
            "mar": round(frame_metrics["mar"], 2),
            "mean_mar_3_frames": round(frame_metrics["mean_mar_3_frames"], 2),
            "yawn_total_frames": periodical_data["yawn_total_frame_count"],
            "yawns_per_minute": round(global_metrics["yawns_per_minute"], 2),
            "prediction": drowsiness_state,
            "yaw": round(frame_metrics["yaw"], 2),
            # "mean_yaw_3_frames": round(frame_metrics["mean_yaw_3_frames"], 2),
            "pitch": round(frame_metrics["pitch"], 2),
            # "mean_first_pitch": round(periodical_data["mean_first_pitch"], 2),
            # "mean_last_pitch": mean_last_pitch,
            # "current_pitch_threshold": round(mean_last_pitch * config["head_nod_threshold_perc"]),
            "head_nod": frame_metrics["head_nod"],
            "num_head_nods": periodical_data["head_nod_count"],
            # "mean_pitch_3_frames": round(frame_metrics["mean_pitch_3_frames"], 2),
            # "mean_nose_tip_y": round(sum(periodical_data["nose_tip_y_values"])/len(periodical_data["nose_tip_y_values"]), 2),
            "nose_tip_y": round(frame_metrics["nose_tip_y"], 2),
            # "mouth_width": round(frame_metrics["mouth_width"], 2),
            # "mean_first_mouth_width": round(periodical_data["mean_first_mouth_width"], 2),
        }

        edited_frame = frame.copy()
        if current_metrics["drawn_frame"] is not None:
            edited_frame = current_metrics["drawn_frame"].copy()
        point = ( int(0), int(0.05 * height) )
        for metric, value in info_to_show.items():
            edited_frame = cv2.putText(edited_frame, f"{metric}: {value}", point, color=(0,255,0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale)
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
    metric_dataframe = pd.DataFrame(metrics)
    metric_dataframe.to_csv(f"{path[:-4]}.csv")
    return predictions


def inference_on_webcam(model, model_features, config, period_length=1):
    cap = cv2.VideoCapture(0)
    max_width = 1080
    frame_metrics = {
        "rear": 0,
        "lear": 0,
    }
    metrics = []
    global_metrics = defaultdict(lambda: 0)
    periodical_data = { 
                        "frame_count" : 0,
                        "closed_eye_frame_count" : 0,
                        "closed_eye_frame_values": [],
                        "current_frames_closed_eyes" : 0,
                        "previous_current_frames_closed_eyes": 0,
                        "max_frames_closed_eyes" : 0,
                        "mean_frames_closed_eyes" : 0,
                        "num_blinks" : 0,
                        "blink_values": [],
                        "previous_frame_eye_state" : None,
                        "previous2_frame_eye_state" : None,
                        "ear_values" : [], 
                        "sum_ear": 0,
                        "sum_first_ear": 0,
                        "sum_left_iris_diameter": 0,
                        "sum_right_iris_diameter": 0,
                        "current_eye_state": None,
                        "head_nod_total_frame_count": 0,
                        "previous_head_nod_state": False,
                        "head_nod_last_frame": -1000,
                        "head_nod_count": 0,
                        "head_nod_values": [],
                        "sum_first_pitch": 0,
                        "pitch_values": [],
                        "yaw_values": [],
                        "yawn_total_frame_count": 0,
                        "previous_yawn_state": False,
                        "yawn_last_frame": -1000,
                        "num_yawns" : 0,
                        "yawn_values": [],
                        "mar_values": [],
                        "nose_tip_y_values": [],
                        "mouth_top_y_values": [],
                        "sum_first_mouth_width": 0,
                        "mean_first_ear": 0,
                       }

    debug = False
    predictions = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    frames_per_minute = int(fps * 60)
    remaining_frames_of_period = period_length

    valid_frame, frame = cap.read()
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter("video_cam_output.avi", fourcc, fps, (width, height))  
    font_scale = min(1, width / max_width)
    metrics_time = 0
    start = time.time()
    while valid_frame:
        remaining_frames_of_period -= 1
        obtain_global_metrics = False
        if remaining_frames_of_period <= 0:
            obtain_global_metrics = True
            remaining_frames_of_period = period_length

        current_metrics = mo.obtain_frame_metrics(frame, periodical_data, config, obtain_global_metrics, fps)
        
        periodical_data = current_metrics["periodical_data"]
        
        # TODO: pasar un parametro o de alguna manera decirle a la funcion que metricas obtener
        if obtain_global_metrics:
            if current_metrics["frame_metrics"] is not None:
                frame_metrics = current_metrics["frame_metrics"]
                global_metrics = current_metrics["global_metrics"]
                selected_metrics = global_metrics

                x_data = np.array(list({ metric:selected_metrics[metric] for metric in model_features }.values())).reshape(1, -1)
                # x_data = np.array(list({ metric:value for metric, value in selected_metrics.items() if metric in model_features }.values())).reshape(1, -1)
                drowsiness_state = model.predict(x_data)
                predictions.append(drowsiness_state)
                            
                flat_metrics = dict(list(current_metrics["global_metrics"].items()) 
                                   + list(current_metrics["frame_metrics"].items()) 
                                   + list(current_metrics["periodical_data"].items()))
                selected_metrics = { metric:value for metric, value in flat_metrics.items() if metric in config["metrics_to_obtain"] }
                metrics.append(selected_metrics)
            else:
                drowsiness_state = -1
                predictions.append(drowsiness_state)
                selected_metrics = {metric:None for metric in config["metrics_to_obtain"]}
                metrics.append(selected_metrics)

        pitch_values_to_analyze = periodical_data["pitch_values"][:-config["num_frames_lag_ignore"]]
        if len(pitch_values_to_analyze) != 0:
            mean_last_pitch = round(sum(pitch_values_to_analyze)/len(pitch_values_to_analyze))
        else:
            mean_last_pitch = -1

        if len(periodical_data["ear_values"]) != 0:
            mean_ear_last_minute = sum(periodical_data["ear_values"])/len(periodical_data["ear_values"])
        else:
            mean_ear_last_minute = 0
        info_to_show = {
            "frame_count": periodical_data["frame_count"],
            # "closed_eye_frame_count": periodical_data["closed_eye_frame_count"],
            # "perclos": round(global_metrics["perclos"], 2),
            # "current_frames_closed_eyes": periodical_data["current_frames_closed_eyes"],
            # "num_blinks": periodical_data["num_blinks"],
            # "blinks_per_minute": round(global_metrics["blinks_per_minute"], 2),
            # "mean_blink_time": round(global_metrics["mean_blink_time"], 2),
            # # "previous_eye_state": periodical_data["previous_frame_eye_state"],
            # # "current_eye_state": periodical_data["current_eye_state"],
            # "ear": round(frame_metrics["ear"], 2),
            # "mean_ear_3_frames": round(frame_metrics["mean_ear_3_frames"], 2),
            # "mean_ear_last_minute": round(mean_ear_last_minute, 2),
            # "mean_EAR": round(periodical_data["mean_first_ear"], 2),
            "num_yawns": periodical_data["num_yawns"],
            "mar": round(frame_metrics["mar"], 2),
            "mean_mar_3_frames": round(frame_metrics["mean_mar_3_frames"], 2),
            "yawn_total_frames": periodical_data["yawn_total_frame_count"],
            "yawns_per_minute": round(global_metrics["yawns_per_minute"], 2),
            "prediction": drowsiness_state,
            "yaw": round(frame_metrics["yaw"], 2),
            "pitch": round(frame_metrics["pitch"], 2),
            "mean_first_pitch": round(periodical_data["mean_first_pitch"], 2),
            "mean_last_pitch": mean_last_pitch,
            "current_pitch_threshold": round(mean_last_pitch * config["head_nod_threshold_perc"]),
            "head_nod": frame_metrics["head_nod"],
            "num_head_nods": periodical_data["head_nod_count"],
            "mean_pitch_3_frames": round(frame_metrics["mean_pitch_3_frames"], 2),
            "mean_nose_tip_y": round(sum(periodical_data["nose_tip_y_values"])/len(periodical_data["nose_tip_y_values"]), 2),
            "nose_tip_y": round(frame_metrics["nose_tip_y"], 2),
            "mouth_width": round(frame_metrics["mouth_width"], 2),
            "mean_first_mouth_width": round(periodical_data["mean_first_mouth_width"], 2),
        }

        edited_frame = frame.copy()
        if current_metrics["drawn_frame"] is not None:
            edited_frame = current_metrics["drawn_frame"].copy()
        point = ( int(0), int(0.05 * height) )
        for metric, value in info_to_show.items():
            edited_frame = cv2.putText(edited_frame, f"{metric}: {value}", point, color=(0,255,0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale)
            point = (point[0], point[1] + int(0.05*height))
        
        out.write(edited_frame)
        cv2.imshow("image", edited_frame)
        start2 = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        valid_frame, frame = cap.read()
        metrics_time += time.time() - start2
        if periodical_data["frame_count"] % 30 == 0:
            end = time.time()
            print(end - start)
            print(metrics_time)
            metrics_time = 0
            start = end
    
    metric_dataframe = pd.DataFrame(metrics)
    metric_dataframe.to_csv("webcam.csv")
    cap.release()
    cv2.destroyAllWindows()