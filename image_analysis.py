import math

import cv2
import numpy as np

eye_closure_threshold = 0.15

def check_eyes_open(img, landmarks, landmark_indexes) -> bool:  
    open_eyes = True
    for eye_name, indexes in landmark_indexes.items():
        eye_closure = compute_eye_closure(img, landmarks, **indexes)
        open_eye = eye_closure >= eye_closure_threshold
        open_eyes = open_eyes and open_eye

    return open_eyes


def compute_eye_closure(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks):
    height, width, _ = img.shape
    upper_landmarks_height = sum([ landmarks.landmark[ind].y * height for ind in upper_landmarks])
    lower_landmarks_height = sum([ landmarks.landmark[ind].y * height for ind in lower_landmarks])
    horizontal_distance = abs(landmarks.landmark[center_landmarks[0]].x - landmarks.landmark[center_landmarks[1]].x) * width

    height = (lower_landmarks_height - upper_landmarks_height)/len(upper_landmarks)
    width = horizontal_distance
    eye_closure = height / width

    print(f"HEIGHT: {height}")
    print(f"WIDTH: {width}")
    print(f"EAR: {eye_closure}")
    print()
    return eye_closure