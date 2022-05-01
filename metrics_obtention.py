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


def process_frame(frame, mean_iris_props=None, framecount=None, config=None):
    faces = roi.mediapipe_face_mesh(frame)

    if faces is None or faces.multi_face_landmarks is None:
        print("didnt find face")
        return None, None
    
    face_landmarks = faces.multi_face_landmarks[0]

    num_landmarks = len(face_landmarks.landmark)
    width_landmarks = [ face_landmarks.landmark[i].x for i in range(0, num_landmarks) ]
    height_landmarks = [ face_landmarks.landmark[i].y for i in range(0, num_landmarks) ]

    ## FACE BOUNDING BOX ################################################################
    if framecount is not None:
        print(f"FRAME: {framecount+1}")
    top = min(height_landmarks)
    bot = max(height_landmarks)
    left = min(width_landmarks)
    right = max(width_landmarks)
    top_ind = height_landmarks.index(top)
    bot_ind = height_landmarks.index(bot)
    right_ind = width_landmarks.index(right)
    left_ind = width_landmarks.index(left)

    width = right - left
    height = bot - top
    indexes = (top_ind, bot_ind, left_ind, right_ind)
    #print([face_landmarks.landmark[ind] for ind in indexes])
    #print(f"width: {width}, height: {height}")
    #####################################################################################
    ## left
    top_left = face_landmarks.landmark[470].y
    bot_left = face_landmarks.landmark[472].y
    iris_diameters_left = (bot_left - top_left)
    iris_prop_left = iris_diameters_left / height

    ## right
    top_right = face_landmarks.landmark[475].y
    bot_right = face_landmarks.landmark[477].y
    iris_diameters_right = (bot_right - top_right)
    iris_prop_right = iris_diameters_right / height
    # print(iris_diameters_left)
    # print(iris_diameters_right)
    
    # print(f"LEFT_EYE_PROP: {iris_prop_left}")
    # print(f"RIGHT_EYE_PROP: {iris_prop_right}")

    if mean_iris_props is None:
        iris_diameters = {"left": iris_diameters_left, "right": iris_diameters_right}
        # print(f"CURR_LEFT_EYE_PROP: {iris_prop_left}")
        # print(f"CURR_RIGHT_EYE_PROP: {iris_prop_right}")
    else:
        iris_diameters = {"left": mean_iris_props["left"]*height, "right": mean_iris_props["right"]*height}
    
        # print(f"CURR_LEFT_EYE_PROP: {mean_iris_props['left']}")
        # print(f"CURR_RIGHT_EYE_PROP: {mean_iris_props['right']}")
    #####################################################################################

    left_eye_indexes = { "upper_landmarks": [158, 159], "lower_landmarks": [144, 145], "center_landmarks": [33, 133] }
    right_eye_indexes = { "upper_landmarks": [386, 385], "lower_landmarks": [374, 380], "center_landmarks": [263, 362] }
    eye_indexes = { "left": left_eye_indexes, "right": right_eye_indexes }

    left_iris_indexes = [ 468, 469, 470, 471, 472 ]
    right_iris_indexes = [ 473, 474, 475, 476, 477 ]
    iris_indexes = { "left": left_iris_indexes, "right": right_iris_indexes}

    upper_lip_indexes = [ 81, 82, 13, 312, 311 ]
    lower_lip_indexes = [ 178, 87, 14, 317, 402 ]
    center_lip_indexes = [ 78, 308 ]
    lip_indexes = { "upper_landmarks": upper_lip_indexes, "lower_landmarks": lower_lip_indexes, "center_landmarks": center_lip_indexes}
    
    #cv2.imshow("", res_img)
    #cv2.waitKey()

    #ROI_images = roi.get_ROI_images(frame, face_landmarks)
    #iris_centers = roi.get_iris_centers(frame, face_landmarks, iris_indexes)

    iris_metrics = roi.get_iris_metrics(frame, face_landmarks, iris_indexes)

    open_eyes = image_analysis.check_eyes_open(frame, face_landmarks, eye_indexes, iris_diameters)
    yawn = image_analysis.check_yawn(frame, face_landmarks, lip_indexes)

    frame_metrics = {}
    # TODO: decidir si se computa la eye_closure como la media de los dos ojos
    frame_metrics["lear"] = image_analysis.compute_eye_closure(frame, face_landmarks, iris_diameter=iris_diameters["left"], **eye_indexes["left"])
    frame_metrics["rear"] = image_analysis.compute_eye_closure(frame, face_landmarks, iris_diameter=iris_diameters["right"], **eye_indexes["right"])
    frame_metrics["open_eyes"] = open_eyes
    frame_metrics["yawn"] = yawn
    frame_metrics["mar"] = image_analysis.compute_mouth_closure(frame, face_landmarks, **lip_indexes)
    frame_metrics["left_iris_diameter"] = iris_prop_left
    frame_metrics["right_iris_diameter"] = iris_prop_right

    drawn_frame = frame.copy()
    indexes = [27, 28, 29, 22, 23, 24, 257, 258, 259, 252, 253, 254]
    #drawn_frame = draw_landmarks(drawn_frame, face_landmarks, indexes)
   # drawn_frame = draw_eye_landmarks(drawn_frame, face_landmarks, eye_indexes)
   # drawn_frame = draw_iris_landmarks(drawn_frame, face_landmarks, iris_indexes)
    return frame_metrics, drawn_frame


def update_periodical_data(frame_metrics: dict, periodical_data: dict) -> dict:
    periodical_data["frame_count"] += 1
    if frame_metrics is None:
        return periodical_data

    if frame_metrics["open_eyes"]:
        periodical_data["current_eye_state"] = "open"
        
        if periodical_data["current_frames_closed_eyes"] > periodical_data["max_frames_closed_eyes"]:
            periodical_data["max_frames_closed_eyes"] = periodical_data["current_frames_closed_eyes"]
        
        periodical_data["current_frames_closed_eyes"] = 0
    else:
        periodical_data["current_eye_state"] = "closed"
        periodical_data["closed_eye_frame_count"] += 1
        periodical_data["current_frames_closed_eyes"] += 1

        if periodical_data["previous_frame_eye_state"] == "open":
            periodical_data["num_blinks"] += 1
    
    if frame_metrics["yawn"]:
        periodical_data["num_yawns"] += 1

    periodical_data["previous_frame_eye_state"] = periodical_data["current_eye_state"]
    #periodical_data["ear_values"].append(frame_metrics["ear"])
    periodical_data["sum_ear"] += frame_metrics["lear"]

    if periodical_data["frame_count"] <= 10:
        periodical_data["sum_left_iris_diameter"] += frame_metrics["left_iris_diameter"]
        periodical_data["sum_right_iris_diameter"] += frame_metrics["right_iris_diameter"]
        periodical_data["mean_left_iris_diameter"] = periodical_data["sum_left_iris_diameter"] / min(10, periodical_data["frame_count"])
        periodical_data["mean_right_iris_diameter"] = periodical_data["sum_right_iris_diameter"] / min(10, periodical_data["frame_count"])

    return periodical_data

def compute_global_metrics(frame_metrics: dict, periodical_data: dict, fps: int, frames_per_minute: int) -> dict:
    global_metrics = {}
    
    global_metrics["mean_ear"] = periodical_data["sum_ear"] / periodical_data["frame_count"]
    global_metrics["blink_frequency"] = periodical_data["num_blinks"] / periodical_data["frame_count"]
    global_metrics["blinks_per_minute"] = periodical_data["num_blinks"] * frames_per_minute / periodical_data["frame_count"]
    global_metrics["perclos"] = periodical_data["closed_eye_frame_count"] / periodical_data["frame_count"]
    global_metrics["current_time_closed_eyes"] = periodical_data["current_frames_closed_eyes"] / fps
    global_metrics["yawns_per_minute"] = periodical_data["num_yawns"] * frames_per_minute / periodical_data["frame_count"]

    return global_metrics


def obtain_frame_metrics(frame, periodical_data, obtain_global_metrics=False, fps=0):

    metrics = {}
    iris_diameters = None
    if "mean_left_iris_diameter" in periodical_data:
        iris_diameters = {"left": periodical_data["mean_left_iris_diameter"], "right": periodical_data["mean_right_iris_diameter"]}

    frame_metrics, _ = process_frame(frame, iris_diameters)
    metrics["frame_metrics"] = frame_metrics
    # TODO: periodical data que tenga en cuenta info de los ultimos x minutos
    periodical_data = update_periodical_data(frame_metrics, periodical_data)
    metrics["periodical_data"] = periodical_data
    
    if frame_metrics is not None:
        if obtain_global_metrics:
            global_metrics = compute_global_metrics(frame_metrics, periodical_data, fps, int(60 * fps))
            global_metrics["frame"] = periodical_data["frame_count"] - 1
            metrics["global_metrics"] = global_metrics

    return metrics

def obtain_metrics_from_video(input_video, subject, config):
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

    metrics = []
    remaining_frames_of_period = config["period_length"]
    fps = round(input_video.get(cv2.CAP_PROP_FPS))
    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame:
        remaining_frames_of_period -= 1
        obtain_global_metrics = False
        if remaining_frames_of_period <= 0:
            obtain_global_metrics = True
            remaining_frames_of_period = config["period_length"]

        current_metrics = obtain_frame_metrics(frame, periodical_data, obtain_global_metrics, fps)
        periodical_data = current_metrics["periodical_data"]

        if obtain_global_metrics and current_metrics["frame_metrics"] is not None:
            flat_metrics = current_metrics["global_metrics"] | current_metrics["frame_metrics"] | current_metrics["periodical_data"]
            selected_metrics = { metric:value for metric, value in flat_metrics.items() if metric in config["metrics_to_obtain"] }

            metrics.append(selected_metrics)

        valid_frame, frame = input_video.read()
        if periodical_data["frame_count"] % 1000 == 0:
            print(f"{periodical_data['frame_count']}: {time.time() - start}")

    return metrics


def create_dataset_from_video(input_video, subject, config, label):

    metric_list = obtain_metrics_from_video(input_video, subject, config)
    metric_dataframe = pd.DataFrame(metric_list)

    fps = round(input_video.get(cv2.CAP_PROP_FPS))
    num_samples = len(metric_list)

    metric_dataframe["label"] = [label] * num_samples
    metric_dataframe["fps"] = [fps] * num_samples
    metric_dataframe["subject"] = [subject] * num_samples
    return metric_dataframe


def create_dataset_from_videos(path, target_folder, config) -> list:
    df_list = []
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        video_extensions = [ ".mp4", ".mov", ".avi", ".mp3" ]

        if os.path.isdir(file):
            df_list = df_list + create_dataset_from_videos(file)
        elif os.path.isfile(file) and filename[-4:].lower() in video_extensions:
            video = cv2.VideoCapture(file)
            print(file)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            subject = file.split('/')[-2]
            print(subject)
            if filename[0] == "0":
                label = 0
                df = create_dataset_from_video(video, subject, config, label)
                df.to_csv(f"{target_folder}{subject}_{label}.csv")
                df_list.append(df)
            elif filename[0] == "1":
                label = 10
                df = create_dataset_from_video(video, subject, config, label)             
                suffix = ""
                if filename[-5] != "0":
                    suffix = f"_{filename[-5]}"

                df.to_csv(f"{target_folder}{subject}_{label}{suffix}.csv")
                df_list.append(df)

    return df_list