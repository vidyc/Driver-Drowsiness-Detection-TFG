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

input_type = "video"
max_num_frames = 5400

model = load("lgb_model.joblib")

def draw_eye_landmarks(img, face_landmarks, eye_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for eye, indexes in eye_indexes.items():
        for eye_pos, landmarks in indexes.items():
            for ind in landmarks:
                point = face_landmarks.landmark[ind]
                point = (int(point.x*width), int(point.y*height))
                res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
                #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)

    return res_img

def draw_iris_landmarks(img, face_landmarks, iris_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for iris, indexes in iris_indexes.items():
        for ind in indexes:
            point = face_landmarks.landmark[ind]
            point = (int(point.x*width), int(point.y*height))
            res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
            #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)

    return res_img

def draw_landmarks(img, face_landmarks, indexes):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for ind in indexes:
        point = face_landmarks.landmark[ind]
        point = (int(point.x*width), int(point.y*height))
        res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
        res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)

    return res_img

def compute_drowsiness_state(frame_metrics, periodical_data, global_metrics: dict, fps) -> dict:
    x_data = np.array([ global_metrics["perclos"], global_metrics["blink_frequency"], global_metrics["current_time_closed_eyes"] ])
    x_data = x_data.reshape(1, -1)
    prediction = model.predict(x_data)[0]
    if prediction == 1:
        prediction = 10
    return prediction

def score(x_data, y_data):
    predictions = []
    num_hits = 0
    num_samples = len(y_data)

    if num_samples <= 0:
        return 0

    for ind, sample in enumerate(x_data):
        perclos = sample[0]
        blinks_per_min = sample[1]
        current_time_closed_eyes = sample[2]
        pred = 0
        if perclos > 0.15 or blinks_per_min < 10 or current_time_closed_eyes > 0.5:
            pred = 1
        
        predictions.append(pred)
        if pred == y_data[ind]:
            num_hits += 1

    return num_hits / num_samples
        
    

def inference_on_dataset(input_video, df):
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print(fps)
    frames_per_minute = int(fps * 60)

    width  = int(input_video.get(3))   # float `width`
    height = int(input_video.get(4))  # float `height`
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter("output_video.avi", fourcc, fps, (width, height))
    x_data = df[["perclos", "blink_frequency", "current_time_closed_eyes"]]
    pred = [round(p) for p in model.predict(x_data)]
    valid_frame, frame = input_video.read()
    ind = 0
    while valid_frame: # and periodical_data["frame_count"] < max_num_frames:
        drowsiness_state = pred[ind]

        edited_frame = frame.copy()
        point = ( int(0), int(0.05 * height) )
        for metric, value in periodical_data.items():
            edited_frame = cv2.putText(edited_frame, f"{metric}: {value}", point, color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)
            point = (point[0], point[1] + int(0.05*height))
        
        for metric, value in global_metrics.items():
            edited_frame = cv2.putText(edited_frame, f"{metric}: {value}", point, color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)
            point = (point[0], point[1] + int(0.05*height))

            edited_frame = cv2.putText(edited_frame, f"prediction: {drowsiness_state}", point, color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1)
            out.write(edited_frame)
        
        valid_frame, frame = input_video.read()
        ind += 1

    out.release()


def inference_on_video(input_video, path="output_video.avi", max_num_frames=float('inf')):  
    max_width = 1080
    periodical_data = { 
                        "frame_count" : 0,
                        "closed_eye_frame_count" : 0,
                        "current_frames_closed_eyes" : 0,
                        "max_frames_closed_eyes" : 0,
                        "mean_frames_closed_eyes" : 0,
                        "num_blinks" : 0,
                        "num_yawns": 0,
                        "previous_frame_eye_state" : None,
                        #"ear_values" : [], 
                        "sum_ear": 0,
                       }

    debug = False
    predictions = []
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print(fps)
    frames_per_minute = int(fps * 60)

    width  = int(input_video.get(3))   # float `width`
    height = int(input_video.get(4))  # float `height`
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))  

    font_scale = min(1, width / max_width)

    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame and periodical_data["frame_count"] < max_num_frames:
        frame_metrics, drawn_frame = process_frame(frame, framecount=periodical_data["frame_count"])
        global_metrics = {}
        drowsiness_state = None
        if frame_metrics is not None:
            periodical_data = update_periodical_data(frame_metrics, periodical_data)
            global_metrics = compute_global_metrics(frame_metrics, periodical_data, fps, frames_per_minute)
            drowsiness_state = compute_drowsiness_state(frame_metrics, periodical_data, global_metrics, fps)

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

            edited_frame = drawn_frame.copy()
            #edited_frame = frame.copy()
            point = ( int(0), int(0.05 * height) )
            for metric, value in info_to_show.items():
                edited_frame = cv2.putText(edited_frame, f"{metric}: {value}", point, color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale)
                point = (point[0], point[1] + int(0.05*height))

            out.write(edited_frame)
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

    out.release()
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
                        "sum_left_iris_diameter": 0,
                        "sum_right_iris_diameter": 0,
                       }

    metrics = []
    remaining_frames_of_period = period_length
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    frames_per_minute = int(fps * 60)
    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame:
        frame_metrics, drawn_frame = process_frame(frame)
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

        ind = 0
        if os.path.isdir(file):
            df_list = df_list + create_dataset_from_videos(file)
        elif os.path.isfile(file) and filename[-4:].lower() in video_extensions:
            video = cv2.VideoCapture(file)
            print(file)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if filename[0] == "0":
                label = 0
                df = create_dataset_from_video(video, label)
                df.to_csv(f"UTA_dataset_pupil/{ind}.csv")
                df_list.append(df)
                ind += 1
            elif filename[0] == "1":
                label = 10
                df = create_dataset_from_video(video, label)
                df.to_csv(f"UTA_dataset_pupil/{ind}.csv")
                df_list.append(df)
                ind += 1

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
