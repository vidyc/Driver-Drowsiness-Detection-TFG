import math
import random
import time
import os
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

import cv2
import pandas as pd
import numpy as np

import region_detection as roi
import image_analysis

input_type = "video"
max_num_frames = 5400

model = load("knn_model.joblib")

def draw_eye_landmarks(img, face_landmarks, eye_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for eye, indexes in eye_indexes.items():
        for eye_pos, landmarks in indexes.items():
            for ind in landmarks:
                point = face_landmarks.landmark[ind]
                point = (int(point.x*width), int(point.y*height))
                res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
                res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)

    return res_img

def draw_iris_landmarks(img, face_landmarks, iris_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for iris, indexes in iris_indexes.items():
        for ind in indexes:
            point = face_landmarks.landmark[ind]
            point = (int(point.x*width), int(point.y*height))
            res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
            res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)

    return res_img

def draw_landmarks(img, face_landmarks, indexes: frozenset):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for source, dest in list(roi.mp_face_mesh.FACEMESH_LIPS):
        point1 = face_landmarks.landmark[source]
        point1 = (int(point1.x * width), int(point1.y * height))
        point2 = face_landmarks.landmark[dest]
        point2 = (int(point2.x * width), int(point2.y * height))
        res_img = cv2.circle(res_img, point1, 2, (255, 0, 0), -1)
        res_img = cv2.putText(res_img, f"{source}", point1, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0))
        res_img = cv2.circle(res_img, point2, 2, (255, 0, 0), -1)
        res_img = cv2.putText(res_img, f"{dest}", point2, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0))

    return res_img

def process_frame(frame, config=None):
    faces = roi.mediapipe_face_mesh(frame)

    if faces is None or faces.multi_face_landmarks is None:
        print("didnt find face")
        return None
    
    face_landmarks = faces.multi_face_landmarks[0]

    left_eye_indexes = { "upper_landmarks": [158, 159], "lower_landmarks": [144, 145], "center_landmarks": [33, 133] }
    right_eye_indexes = { "upper_landmarks": [386, 385], "lower_landmarks": [374, 380], "center_landmarks": [263, 362] }
    eye_indexes = { "left_eye": left_eye_indexes, "right_eye": right_eye_indexes }

    right_iris_indexes = [ 468, 469, 470, 471, 472 ]
    left_iris_indexes = [ 473, 474, 475, 476, 477 ]
    iris_indexes = { "left_iris": left_iris_indexes, "right_iris": right_iris_indexes}

    upper_lip_indexes = [ 81, 82, 13, 312, 311 ]
    lower_lip_indexes = [ 178, 87, 14, 317, 402 ]
    center_lip_indexes = [ 78, 308 ]
    lip_indexes = { "upper_landmarks": upper_lip_indexes, "lower_landmarks": lower_lip_indexes, "center_landmarks": center_lip_indexes}
    
    #cv2.imshow("", res_img)
    #cv2.waitKey()

    #ROI_images = roi.get_ROI_images(frame, face_landmarks)
    #iris_centers = roi.get_iris_centers(frame, face_landmarks, iris_indexes)

    open_eyes = image_analysis.check_eyes_open(frame, face_landmarks, eye_indexes)
    yawn = image_analysis.check_yawn(frame, face_landmarks, lip_indexes)

    frame_metrics = {}
    # TODO: decidir si se computa la eye_closure como la media de los dos ojos
    frame_metrics["ear"] = image_analysis.compute_eye_closure(frame, face_landmarks, **eye_indexes["left_eye"])
    frame_metrics["open_eyes"] = open_eyes
    frame_metrics["yawn"] = yawn
    frame_metrics["mar"] = image_analysis.compute_mouth_closure(frame, face_landmarks, **lip_indexes)

    return frame_metrics


def update_periodical_data(frame_metrics: dict, periodical_data: dict) -> dict:
    periodical_data["frame_count"] += 1

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
    periodical_data["sum_ear"] += frame_metrics["ear"]

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

def compute_drowsiness_state(frame_metrics: dict, periodical_data: dict, global_metrics: dict, fps: int) -> dict:
    x_data = np.array([ global_metrics["blink_frequency"], global_metrics["perclos"], global_metrics["current_time_closed_eyes"] ])
    x_data = x_data.reshape(1, -1)
    prediction = model.predict(x_data)[0]
    if prediction == 1:
        prediction = 10
    return prediction


def inference_on_video(input_video):    
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
                       }

    debug = False
    predictions = []
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print(fps)
    frames_per_minute = int(fps * 60)
    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame: # and periodical_data["frame_count"] < max_num_frames:
        frame_metrics = process_frame(frame)
        global_metrics = {}
        drowsiness_state = None
        if frame_metrics is not None:
            periodical_data = update_periodical_data(frame_metrics, periodical_data)
            global_metrics = compute_global_metrics(frame_metrics, periodical_data, fps, frames_per_minute)
            drowsiness_state = compute_drowsiness_state(frame_metrics, periodical_data, global_metrics, fps)
        
        predictions.append(drowsiness_state)
        
        valid_frame, frame = input_video.read()
        
        if periodical_data["frame_count"] % 1000 == 0:
            print(f"{periodical_data['frame_count']}: {time.time() - start}")
            print(periodical_data)
            print(global_metrics)
            print(drowsiness_state)

        if debug:
            cv2.imshow('', frame)
            cv2.waitKey(0)

    
    print(frame_metrics)
    print()
    print(periodical_data)
    print()
    print(global_metrics)

    return predictions


def obtain_metrics_from_video(input_video, period_length=1):
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
                       }

    metrics = []
    remaining_frames_of_period = period_length
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frames_per_minute = int(fps * 60)
    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame:
        frame_metrics = process_frame(frame)
        global_metrics = None
        if frame_metrics is not None:
            global_metrics = {}
            # TODO: periodical data que tenga en cuenta info de los ultimos x minutos
            periodical_data = update_periodical_data(frame_metrics, periodical_data)
            remaining_frames_of_period -= 1

            if remaining_frames_of_period <= 0:
                global_metrics = compute_global_metrics(frame_metrics, periodical_data, fps, frames_per_minute)
                remaining_frames_of_period = period_length
                global_metrics["frame"] = periodical_data["frame_count"] - 1
                metrics.append(global_metrics)
        
        valid_frame, frame = input_video.read()
        if periodical_data["frame_count"] % 1000 == 0:
            print(f"{periodical_data['frame_count']}: {time.time() - start}")

    return metrics


def create_dataset_from_video(input_video, label):

    metric_list = obtain_metrics_from_video(input_video)
    metric_dataframe = pd.DataFrame(metric_list)

    labels = [label] * len(metric_list)
    metric_dataframe["label"] = labels

    return metric_dataframe


def create_dataset_from_videos(path) -> list:
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
            if filename[0] == "0":
                label = 0
                df_list.append(create_dataset_from_video(video, label))
            elif filename[0] == "1":
                label = 10
                df_list.append(create_dataset_from_video(video, label))
    
    #df = pd.concat(df_list)
    return df_list


def euclidean_distance(point1, point2):
    point = point1[1:-1]
    sum_squared_distance = sum(math.pow(point[i] - point2[i], 2) for i in range(len(point)))

    return math.sqrt(sum_squared_distance)

def train_knn_model(df, k):
    # un test con el sujeto 31 de fold3...
    # procesamos los videos 0 y 10 --> obtenemos un 90% de los frames =? 30k frames * 3 metricas = 90k metricas
    # para cada uno de los 35k frames tenemos un label --> 0 o 10
    # testeamos el rendimiento con el 10% que no hemos cogido
    data = df.drop("label", axis = 1)
    labels = df["label"]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=1 )
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    return knn
