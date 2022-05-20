import math
from ssl import cert_time_to_seconds

import cv2
import numpy as np
from requests import head

cm_in_pixel = 0.0264583333

def pix_to_cm(pixels):
    return pixels * cm_in_pixel

def check_eyes_open(ear, eye_closure_threshold, gray_zone_perc, previous_eye_state) -> bool:  
    
    upper_threshold = eye_closure_threshold * (1 + gray_zone_perc)
    lower_threshold = eye_closure_threshold * (1 - gray_zone_perc)

    if ear < lower_threshold:
        open_eye = False
    elif ear > upper_threshold:
        open_eye = True
    else:
        open_eye = ear >= eye_closure_threshold
        if previous_eye_state is not None:
            open_eye = previous_eye_state

    return open_eye


def check_yawn(mar, mouth_yawn_threshold) -> bool:
    return mar >= mouth_yawn_threshold


def check_head_nod(pitch, head_nod_threshold) -> bool: 

    ####
    # quiero que cuando se detecte un nod no se pueda detectar otro en X frames
    # ademas, quiero que si se pasa mas de X frames con la cabeza nodeada, solo cuente ese.
    # ideas: 
    # - doble threshold? --> no tiene mucho sentido en este caso
    # -
    ####

    return pitch <= head_nod_threshold


def compute_eye_closure1(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks, iris_diameter):
    height, width, _ = img.shape
    upper_landmarks_height = sum(landmarks[ind].y * height for ind in upper_landmarks)
    lower_landmarks_height = sum(landmarks[ind].y * height for ind in lower_landmarks)
    horizontal_distance = abs(landmarks[center_landmarks[0]].x - landmarks[center_landmarks[1]].x) * width
    
    height = (lower_landmarks_height - upper_landmarks_height)/len(upper_landmarks)
    width = horizontal_distance

    eye_closure = height / width
    return eye_closure

def compute_eye_closure2(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks, iris_diameter):
    upper_landmarks = [ np.array([landmarks[ind].x, landmarks[ind].y, landmarks[ind].z ]) for ind in upper_landmarks ]
    lower_landmarks = [ np.array([landmarks[ind].x, landmarks[ind].y, landmarks[ind].z ]) for ind in lower_landmarks ]
    center_landmarks = [ np.array([landmarks[ind].x, landmarks[ind].y, landmarks[ind].z ]) for ind in center_landmarks ]

    length = len(upper_landmarks)
    sum_of_vert_distances = 0
    for ind in range(0, length):
        sum_of_vert_distances += np.linalg.norm(upper_landmarks[ind] - lower_landmarks[ind])

    sum_of_vert_distances /= length
    horizontal_distance = np.linalg.norm(center_landmarks[0] - center_landmarks[1])
    eye_closure = sum_of_vert_distances / horizontal_distance
    return eye_closure

def compute_eye_closure3(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks, iris_diameter):
    height, width, _ = img.shape
    upper_landmarks_height = sum(landmarks[ind].y for ind in upper_landmarks)
    lower_landmarks_height = sum(landmarks[ind].y for ind in lower_landmarks)
    
    horizontal_distance = abs(landmarks[center_landmarks[0]].x - landmarks[center_landmarks[1]].x) * width
    height = (lower_landmarks_height - upper_landmarks_height)/len(upper_landmarks)

    eye_closure = min(1, height / iris_diameter)
    return eye_closure


def compute_mouth_closure(img, landmarks, upper_landmarks, lower_landmarks, center_landmarks):
    
    height, width, _ = img.shape
    
    # FIRST METHOD
    upper_landmarks_height = sum(landmarks[ind].y * height for ind in upper_landmarks)
    lower_landmarks_height = sum(landmarks[ind].y * height for ind in lower_landmarks)
    horizontal_distance = abs(landmarks[center_landmarks[0]].x - landmarks[center_landmarks[1]].x) * width

    height = (lower_landmarks_height - upper_landmarks_height)/len(upper_landmarks)
    width = horizontal_distance
    mouth_closure = height / width

    # SECOND METHOD
    upper_landmarks = [ np.array([landmarks[ind].x, landmarks[ind].y]) for ind in upper_landmarks ]
    lower_landmarks = [ np.array([landmarks[ind].x, landmarks[ind].y]) for ind in lower_landmarks ]
    center_landmarks = [ np.array([landmarks[ind].x, landmarks[ind].y]) for ind in center_landmarks ]

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