import math
from ssl import cert_time_to_seconds

import cv2
import numpy as np

eye_closure_threshold = 0.15
mouth_yawn_threshold = 0.45

def check_eyes_open(img, landmarks, landmark_indexes) -> bool:  
    open_eyes = True
    for eye_name, indexes in landmark_indexes.items():
        eye_closure = compute_eye_closure(img, landmarks, **indexes)
        open_eye = eye_closure >= eye_closure_threshold
        open_eyes = open_eyes and open_eye

    return open_eyes


def check_yawn(img, landmarks, landmark_indexes) -> bool:
    mouth_height = compute_mouth_closure(img, landmarks, **landmark_indexes)
    return mouth_height >= mouth_yawn_threshold


def compute_eye_closure(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks):
    height, width, _ = img.shape
    upper_landmarks_height = sum(landmarks.landmark[ind].y * height for ind in upper_landmarks)
    lower_landmarks_height = sum(landmarks.landmark[ind].y * height for ind in lower_landmarks)
    horizontal_distance = abs(landmarks.landmark[center_landmarks[0]].x - landmarks.landmark[center_landmarks[1]].x) * width

    height = (lower_landmarks_height - upper_landmarks_height)/len(upper_landmarks)
    width = horizontal_distance
    eye_closure = height / width

    return eye_closure


def compute_mouth_closure(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks):
    
    height, width, _ = img.shape
    
    # FIRST METHOD
    upper_landmarks_height = sum(landmarks.landmark[ind].y * height for ind in upper_landmarks)
    lower_landmarks_height = sum(landmarks.landmark[ind].y * height for ind in lower_landmarks)
    horizontal_distance = abs(landmarks.landmark[center_landmarks[0]].x - landmarks.landmark[center_landmarks[1]].x) * width

    height = (lower_landmarks_height - upper_landmarks_height)/len(upper_landmarks)
    width = horizontal_distance
    mouth_closure = height / width

    # SECOND METHOD
    upper_landmarks = [ np.array([landmarks.landmark[ind].x, landmarks.landmark[ind].y, landmarks.landmark[ind].z ]) for ind in upper_landmarks ]
    lower_landmarks = [ np.array([landmarks.landmark[ind].x, landmarks.landmark[ind].y, landmarks.landmark[ind].z ]) for ind in lower_landmarks ]
    center_landmarks = [ np.array([landmarks.landmark[ind].x, landmarks.landmark[ind].y, landmarks.landmark[ind].z ]) for ind in center_landmarks ]

    length = len(upper_landmarks)
    sum_of_vert_distances = 0
    for ind in range(0, length):
        sum_of_vert_distances += np.linalg.norm(upper_landmarks[ind] - lower_landmarks[ind])

    sum_of_vert_distances /= length
    horizontal_distance = np.linalg.norm(center_landmarks[0] - center_landmarks[1])
    mouth_closure = sum_of_vert_distances / horizontal_distance

    # TODO: probar dos metodos
    # 1 - distancias euclideanas, a ver si asi evitamos un poco los casos con enfoque no frontal
    # 2 - approach para videos, utilizar los primeros frames del video para determinar la distancia cuando la boca esta cerrada y trabajar con eso

    return mouth_closure