import math
import random
import time
import copy
import os
from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np

import cv2
import pandas as pd
# import tensorflow as tf

import region_detection as roi
import image_analysis

periodical_data_initial_state = { 
                    "frame_count" : 0,
                    "closed_eye_frame_count" : 0,
                    "closed_eye_frame_values": [],
                    "current_frames_closed_eyes" : 0,
                    "previous_current_frames_closed_eyes": 0,
                    "max_frames_closed_eyes" : 0,
                    "mean_frames_closed_eyes" : 0,
                    "num_blinks" : 0,
                    "blink_values": [],
                    "previous_frame_mouth_state": None,
                    "previous_frame_eye_state" : None,
                    "previous2_frame_eye_state" : None,
                    "ear_values": [], 
                    "open_ear_values": [],
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
                    "mean_first_pitch": None,
                    "left_iris_prop": [],
                    "right_iris_prop": [],
                    "left_euclidean_iris_prop": [],
                    "right_euclidean_iris_prop": [],
                    "expected_ear": 0,
                    }


# tf_model = tf.keras.models.load_model('saved_model7/my_model')
def tensorflow_open_close(filenames):

    for filename in filenames:
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)
        image = tf.image.resize(image, (160, 160))
        image = tf.reshape(image, [1, 160, 160, 3])

        predictions = tf_model.predict(image)

        predictions = tf.nn.sigmoid(predictions)
        print('Predictions_logit:\n', predictions.numpy())
        predictions = tf.where(predictions < 0.5, 0, 1)

        print('Predictions:\n', predictions.numpy())

def draw_landmarks(img, face_landmarks, indexes):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for ind in indexes:
        point = face_landmarks[ind]
        point = (int(point.x*width), int(point.y*height))
        res_img = cv2.circle(res_img, point, radius=2, color=(0, 0, 255), thickness=-1)
        res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def draw_eye_landmarks(img, face_landmarks, eye_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for eye, indexes in eye_indexes.items():
        for eye_pos, landmarks in indexes.items():
            for ind in landmarks:
                point = face_landmarks[ind]
                point = (int(point.x*width), int(point.y*height))
                res_img = cv2.circle(res_img, point, radius=2, color=(0, 0, 255), thickness=-1)
                #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def draw_lip_landmarks(img, face_landmarks, lip_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for lip_pos, landmarks in lip_indexes.items():
        for ind in landmarks:
            point = face_landmarks[ind]
            point = (int(point.x*width), int(point.y*height))
            res_img = cv2.circle(res_img, point, radius=2, color=(0, 0, 255), thickness=-1)
            #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def draw_iris_landmarks(img, face_landmarks, iris_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for iris, indexes in iris_indexes.items():
        for ind in indexes:
            point = face_landmarks[ind]
            point = (int(point.x*width), int(point.y*height))
            res_img = cv2.circle(res_img, point, radius=2, color=(255, 255, 0), thickness=-1)
            #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def process_frame(frame, config, eyes=True, mouth=True, head=True, expected_ear=None, last_pose_angles=None, last_iris_prop_values=None, mean_first_ear=None, mean_first_pitch=None, last_pitch_values=None, last_yaw_values=None, last_mar_values=None, last_ear_values=None, nose_tip_y_values=None, mouth_top_y_values=None, previous_eye_state=None, previous_yawn_state=None, framecount=None):
    frame_height, frame_width, _ = frame.shape
    # roi.opencv_detect_faces(frame)

    # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # lab_planes = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # lab_planes_0 = clahe.apply(lab_planes[0])
    # lab_planes = (lab_planes_0, *lab_planes[1:])
    # lab = cv2.merge(lab_planes)
    # clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    # mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    # frame = cv2.inpaint(frame, mask2, 0.1, cv2.INPAINT_TELEA) 

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(cv2.equalizeHist(gray_frame), cv2.COLOR_GRAY2BGR)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # frame = cv2.cvtColor(clahe.apply(gray_frame), cv2.COLOR_GRAY2BGR)
    
    alpha = 1.3 # Simple contrast control
    beta = 30    # Simple brightness control
    # gamma = 0.8
    # # cv2.imshow('Original Image', frame)
    # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    # # cv2.imshow('New Image', frame)   
    # # lookUpTable = np.empty((1,256), np.uint8)
    # # for i in range(256):
    # #     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    # # frame = cv2.LUT(frame, lookUpTable)

    # # Show stuff
    # cv2.imshow('Gamma Image', frame)
    # # Wait until user press some key
    # cv2.waitKey()

    # drawn_frame = frame.copy()
    # drawn_frame = roi.dlib_face_landmarks(frame, drawn_frame)
    faces, frame = roi.mediapipe_face_mesh(frame, debug=False)
    drawn_frame = frame.copy()

    dnn = False
    if faces is None or faces.multi_face_landmarks is None:
        # print("Mediapipe didnt find face. Trying with dnn...")
        # print(framecount)
        return None, drawn_frame, None
        found_face, facebox, drawn_frame = roi.dnn_face_detection(frame, frame.copy())

        if not found_face:
            # print("DNN didnt find face either.")
            return None, drawn_frame, None

        dnn = True
        face_landmarks, drawn_frame = roi.estimate_alternative_landmarks(frame, facebox, drawn_frame, debug=False)
    else:
        face_landmarks = faces.multi_face_landmarks[0].landmark
    
    # TODO: normalizar variable face_landmarks para los dos posibles casos que tenemos

    if dnn:
        return None, drawn_frame, None
    else:
        ## FACE BOUNDING BOX ################################################################
        num_landmarks = len(face_landmarks)
        width_landmarks = [ face_landmarks[i].x for i in range(0, num_landmarks) ]
        height_landmarks = [ face_landmarks[i].y for i in range(0, num_landmarks) ]
        top = min(height_landmarks)
        bot = max(height_landmarks)
        left = min(width_landmarks)
        right = max(width_landmarks)
        top_ind = height_landmarks.index(top)
        bot_ind = height_landmarks.index(bot)
        right_ind = width_landmarks.index(right)
        left_ind = width_landmarks.index(left)

        x = int(left*frame_width)
        y = int(top*frame_height)
        x1 = int(right*frame_width)
        y1 = int(bot*frame_height)

        width = right - left
        height = bot - top
        face_dimensions = (width * frame_width, height * frame_height)
        indexes = (top_ind, bot_ind, left_ind, right_ind)
        #print([face_landmarks[ind] for ind in indexes])
        #print(f"width: {width}, height: {height}")

        # ROI_images = roi.get_ROI_images(frame, face_dimensions, face_landmarks)
        ROI_images = []
        #####################################################################################

        nose_tip_y_diff_threshold = config["head_nod_y_min_threshold"]
        mouth_top_y_diff_threshold = config["yawn_y_min_threshold"]

        head_nod_threshold = config["head_nod_threshold"]
        head_nod_threshold_perc = config["head_nod_threshold_perc"]

        yawn_threshold = config["mouth_yawn_threshold"]

        threshold = config["eye_closure_threshold"]

        main_threshold_perc = config["main_threshold_perc"]
        gray_zone_perc = config["gray_zone_perc"]
        frame_metrics = {}

        if eyes:
            ## right
            top_right= face_landmarks[470].y
            bot_right = face_landmarks[472].y
            iris_right_top = np.array([face_landmarks[470].x, face_landmarks[470].y])
            iris_right_bottom = np.array([face_landmarks[472].x, face_landmarks[472].y])
            
            iris_diameters_right = (bot_right - top_right)
            iris_euclidean_diameters_right = np.linalg.norm(iris_right_bottom - iris_right_top)
            iris_prop_right = iris_diameters_right / height
            iris_prop_euclidean_right = iris_euclidean_diameters_right / height

            ## left
            top_left = face_landmarks[475].y
            bot_left = face_landmarks[477].y
            iris_left_top = np.array([face_landmarks[475].x, face_landmarks[475].y])
            iris_left_bottom = np.array([face_landmarks[477].x, face_landmarks[477].y])

            iris_diameters_left = (bot_left - top_left)
            iris_euclidean_diameters_left = np.linalg.norm(iris_left_bottom - iris_left_top)
            iris_prop_left = iris_diameters_left / height
            iris_prop_euclidean_left = iris_euclidean_diameters_left / height
        
            # right_eye_indexes = { "upper_landmarks": [56, 28, 27, 29, 30], "lower_landmarks": [26, 22, 23, 24, 110], "center_landmarks": [130, 243] }
            # left_eye_indexes = { "upper_landmarks": [286, 258, 257, 259, 260], "lower_landmarks": [256, 252, 253, 254, 339], "center_landmarks": [359, 463] }

            # right_eye_indexes = { "upper_landmarks": [157, 158, 159, 160, 161], "lower_landmarks": [154, 153, 145, 144, 163], "center_landmarks": [33, 133] }
            # left_eye_indexes = { "upper_landmarks": [384, 385, 386, 387, 388], "lower_landmarks": [381, 380, 374, 373, 390], "center_landmarks": [263, 362] }
            right_eye_indexes = { "upper_landmarks": [158, 159, 160], "lower_landmarks": [153, 145, 144], "center_landmarks": [33, 133] }
            left_eye_indexes = { "upper_landmarks": [385, 386, 387], "lower_landmarks": [380, 374, 373], "center_landmarks": [263, 362] }
            eye_indexes = { "left": left_eye_indexes, "right": right_eye_indexes }

            # print(f"LEFT_EYE_LANDMARKS")
            left_sum_z = 0
            num_left_landmarks = 0
            for pos, indexes in left_eye_indexes.items():
                for index in indexes:
                    left_sum_z += face_landmarks[index].z
                    num_left_landmarks += 1
            
            right_sum_z = 0
            num_right_landmarks = 0
            # print(f"RIGHT_EYE_LANDMARKS")
            for pos, indexes in right_eye_indexes.items():
                for index in indexes:
                    right_sum_z += face_landmarks[index].z
                    num_right_landmarks += 1
            
            mean_left_z = left_sum_z / num_left_landmarks
            mean_right_z = right_sum_z / num_right_landmarks

            # print(f"MEAN RIGHT Z: {mean_right_z}")
            # print(f"MEAN LEFT Z: {mean_left_z}")
            
            closer_eye = "right"
            closer_eye_indexes = right_eye_indexes
            if mean_left_z < mean_right_z:
                closer_eye = "left"
                closer_eye_indexes = left_eye_indexes
            
            # print(f"CLOSER EYE: {closer_eye}")

            right_iris_indexes = [ 468, 469, 470, 471, 472 ]
            left_iris_indexes = [ 473, 474, 475, 476, 477 ]
            iris_indexes = { "left": left_iris_indexes, "right": right_iris_indexes}

            # iris_centers = roi.get_iris_centers(frame, face_landmarks, iris_indexes)
            # iris_metrics = roi.get_iris_metrics(frame, face_landmarks, iris_indexes)
            method = config["eye_closure_method"]
            if method == 1:
                eye_closure_func = image_analysis.compute_eye_closure1
            elif method == 2:
                eye_closure_func = image_analysis.compute_eye_closure2
            elif method == 3:
                eye_closure_func = image_analysis.compute_eye_closure3
            else:
                eye_closure_func = image_analysis.compute_eye_closure4

            iris_diameters = {"left": iris_diameters_left, "right": iris_diameters_right}
            iris_euclidean_diameters = {"left": iris_euclidean_diameters_left, "right": iris_euclidean_diameters_right}
            if last_iris_prop_values is not None:
                
                left_iris_props = last_iris_prop_values["left"]
                mean_left_iris_prop = (sum(left_iris_props) + iris_prop_left)/(len(left_iris_props) + 1)
                mean_left_iris_diameter = mean_left_iris_prop * height

                right_iris_props = last_iris_prop_values["right"]
                mean_right_iris_prop = (sum(right_iris_props) + iris_prop_right)/(len(right_iris_props) + 1)
                mean_right_iris_diameter = mean_right_iris_prop * height
                
                left_euclidean_iris_props = last_iris_prop_values["left_euclidean"]
                mean_left_euclidean_iris_prop = (sum(left_euclidean_iris_props) + iris_prop_euclidean_left)/(len(left_euclidean_iris_props) + 1)
                mean_left_euclidean_iris_diameter = mean_left_euclidean_iris_prop * height
                
                right_euclidean_iris_props = last_iris_prop_values["right_euclidean"]
                mean_right_euclidean_iris_prop = (sum(right_euclidean_iris_props) + iris_prop_euclidean_right)/(len(right_euclidean_iris_props) + 1)
                mean_right_euclidean_iris_diameter = mean_right_euclidean_iris_prop * height

                iris_diameters = {"left": mean_left_iris_diameter, "right": mean_right_iris_diameter}
                iris_euclidean_diameters = {"left": mean_left_euclidean_iris_diameter, "right": mean_right_euclidean_iris_diameter}
                
            method_iris_diameters = iris_diameters
            if method == 4:
                method_iris_diameters = iris_euclidean_diameters

            if mean_first_ear is not None:
                threshold = mean_first_ear * main_threshold_perc
        
            if last_ear_values is not None:
                threshold = main_threshold_perc * sum(last_ear_values) / len(last_ear_values)

            gray_zone = 0
            if expected_ear != 0:
                threshold = expected_ear * main_threshold_perc
                gray_zone = expected_ear * gray_zone_perc

            ear = eye_closure_func(frame, face_landmarks, iris_diameter=method_iris_diameters[closer_eye], **eye_indexes[closer_eye])
            ear1 = image_analysis.compute_eye_closure1(frame, face_landmarks, iris_diameter=iris_diameters[closer_eye], **eye_indexes[closer_eye])
            ear2 = image_analysis.compute_eye_closure2(frame, face_landmarks, iris_diameter=iris_diameters[closer_eye], **eye_indexes[closer_eye])
            ear3 = image_analysis.compute_eye_closure3(frame, face_landmarks, iris_diameter=iris_diameters[closer_eye], **eye_indexes[closer_eye]) 
            ear4 = image_analysis.compute_eye_closure4(frame, face_landmarks, iris_diameter=iris_euclidean_diameters[closer_eye], **eye_indexes[closer_eye]) 
            mean_ear = ear
            # if last_ear_values is not None:
            #     ear_last = min(2, len(last_ear_values))
            #     ear_values_to_analyze = last_ear_values[-ear_last:]
            #     mean_ear = (sum(ear_values_to_analyze) + ear) / (ear_last + 1)
        
            open_eyes = image_analysis.check_eyes_open(mean_ear, threshold, gray_zone, previous_eye_state)
            drawn_frame = draw_eye_landmarks(drawn_frame, face_landmarks, {"eye": closer_eye_indexes})
            drawn_frame = draw_iris_landmarks(drawn_frame, face_landmarks, {"iris": iris_indexes[closer_eye]})
        else:
            ear = 0
            ear1 = 0
            ear2 = 0
            ear3 = 0
            ear4 = 0
            mean_ear = 0
            open_eyes = None
            iris_prop_left = 0
            iris_prop_right = 0
            iris_prop_euclidean_left = 0
            iris_prop_euclidean_right = 0

        if mouth:
            # upper_lip_indexes = [ 81, 82, 13, 312, 311 ]
            # lower_lip_indexes = [ 178, 87, 14, 317, 402 ]
            upper_lip_indexes = [ 82, 13, 312 ]
            lower_lip_indexes = [ 87, 14, 317 ]
            center_lip_indexes = [ 78, 308 ]
            lip_indexes = { "upper_landmarks": upper_lip_indexes, "lower_landmarks": lower_lip_indexes, "center_landmarks": center_lip_indexes}
        
            mouth_top_y = (face_landmarks[17].y - face_landmarks[0].y) / height # dividimos entre la altura de la cabeza
            
            right_mouth = np.array([face_landmarks[308].x, face_landmarks[308].y])
            left_mouth = np.array([face_landmarks[78].x, face_landmarks[78].y])
            mouth_width = (face_landmarks[308].x - face_landmarks[78].x) / width
            # mouth_width = np.linalg.norm(right_mouth - left_mouth)

            method = config["mouth_closure_method"]
            if method == 1:
                mouth_closure_func = image_analysis.compute_mouth_closure1
            elif method == 2:
                mouth_closure_func = image_analysis.compute_mouth_closure2

            mar = mouth_closure_func(frame, face_landmarks, **lip_indexes)
            mar1 = image_analysis.compute_mouth_closure1(frame, face_landmarks, **lip_indexes)
            mar2 = image_analysis.compute_mouth_closure2(frame, face_landmarks, **lip_indexes)
            mean_mar = mar
            if last_mar_values is not None:
                mar_last = min(config["num_past_frames_mouth"], len(last_mar_values))
                mar_values_to_analyze = last_mar_values[-mar_last:]
                mean_mar = (sum(mar_values_to_analyze) + mar) / (mar_last + 1)
            
            if mouth_top_y_values is not None:
                mean_mouth_top_y = sum(mouth_top_y_values) / len(mouth_top_y_values)
                mouth_top_y_threshold = mean_mouth_top_y + mouth_top_y_diff_threshold
                # possible_yawn = mouth_top_y >= mouth_top_y_threshold
            
            yawn = image_analysis.check_yawn(mean_mar, yawn_threshold, config["gray_zone_mouth"], previous_yawn_state)
            drawn_frame = draw_lip_landmarks(drawn_frame, face_landmarks, lip_indexes)
        else:
            mar = 0
            mar1 = 0
            mar2 = 0
            mean_mar = 0
            yawn = None
            mouth_top_y = 0
            mouth_width = 0
        
        if head:
            pose_angle_dict = last_pose_angles
            if last_pose_angles is None:
                pose_angle_dict, drawn_frame = roi.estimate_head_pose_model(frame, drawn_frame, (x, y, x1, y1), debug=False)
            
            pose_angle_dict2, _ = roi.estimate_head_pose(frame, drawn_frame, face_landmarks)
            
            if mean_first_pitch is not None:
                head_nod_threshold = mean_first_pitch * head_nod_threshold_perc

            mean_pitch = pose_angle_dict["pitch"]
            if last_pitch_values is not None:
                pitch_last = min(2, len(last_pitch_values))
                pitch_values_to_analyze = last_pitch_values[-pitch_last:]
                mean_pitch = (sum(pitch_values_to_analyze) + pose_angle_dict["pitch"]) / (pitch_last + 1)

            mean_yaw = pose_angle_dict["yaw"]
            if last_yaw_values is not None:
                yaw_last = min(2, len(last_yaw_values))
                yaw_values_to_analyze = last_yaw_values[-yaw_last:]
                mean_yaw = (sum(yaw_values_to_analyze) + pose_angle_dict["yaw"]) / (yaw_last + 1)

            nose_tip_y = face_landmarks[1].y
            possible_head_nod = not open_eyes
            if not open_eyes and nose_tip_y_values is not None:
                mean_nose_tip_y = sum(nose_tip_y_values) / len(nose_tip_y_values)
                nose_tip_y_threshold = mean_nose_tip_y + nose_tip_y_diff_threshold
                possible_head_nod = nose_tip_y >= nose_tip_y_threshold
            
            # if last_pitch_values is not None and framecount > num_frames_lag_ignore:
            #     pitch_values_to_analyze = last_pitch_values[:-num_frames_lag_ignore]
            #     head_nod_threshold = head_nod_threshold_perc * sum(pitch_values_to_analyze) / len(pitch_values_to_analyze)
            head_nod = possible_head_nod
            if True:
                head_nod = image_analysis.check_head_nod(mean_pitch, head_nod_threshold)
        else:
            pose_angle_dict = {"pitch": 0, "yaw": 0, "roll": 0}
            pose_angle_dict2 = {"pitch": 0, "yaw": 0, "roll": 0}
            mean_yaw = 0
            mean_pitch = 0
            nose_tip_y = 0
            head_nod = None

        frame_metrics["roll"] = pose_angle_dict["roll"]
        frame_metrics["yaw"] = pose_angle_dict["yaw"]
        frame_metrics["yaw1"] = pose_angle_dict["yaw"]
        frame_metrics["yaw2"] = pose_angle_dict2["yaw"]
        frame_metrics["pitch"] = pose_angle_dict["pitch"]
        frame_metrics["pitch1"] = pose_angle_dict["pitch"]
        frame_metrics["pitch2"] = pose_angle_dict2["pitch"]
        frame_metrics["mean_yaw_3_frames"] = mean_yaw
        frame_metrics["mean_pitch_3_frames"] = mean_pitch
        frame_metrics["nose_tip_y"] = nose_tip_y
        frame_metrics["head_nod"] = head_nod

        frame_metrics["ear"] = ear
        frame_metrics["ear1"] = ear1
        frame_metrics["ear2"] = ear2
        frame_metrics["ear3"] = ear3
        frame_metrics["ear4"] = ear4
        frame_metrics["mean_ear_3_frames"] = mean_ear
        frame_metrics["open_eyes"] = open_eyes
        frame_metrics["left_iris_diameter"] = iris_prop_left
        frame_metrics["left_euclidean_iris_diameter"] = iris_prop_euclidean_left
        frame_metrics["right_iris_diameter"] = iris_prop_right
        frame_metrics["right_euclidean_iris_diameter"] = iris_prop_euclidean_right
        
        frame_metrics["mar"] = mar
        frame_metrics["mar1"] = mar1
        frame_metrics["mar2"] = mar2
        frame_metrics["mean_mar_3_frames"] = mean_mar
        frame_metrics["yawn"] = yawn
        frame_metrics["mouth_top_y"] = mouth_top_y
        frame_metrics["mouth_width"] = mouth_width

        #indexes = [27, 28, 29, 22, 23, 24, 257, 258, 259, 252, 253, 254]
        # drawn_frame = draw_iris_landmarks(drawn_frame, face_landmarks, iris_indexes)
        #drawn_frame = cv2.rectangle(drawn_frame, (int(top*frame_height), int(left*frame_width)), (int(bot*frame_height), int(right*frame_width)), (0, 255, 0), 2)
        # cv2.imshow("f", drawn_frame)
        # print(f"CLOSER EYE: {closer_eye}")
        # cv2.imshow("", ROI_images[f"{closer_eye}_eye"])
        # cv2.waitKey()
    return frame_metrics, drawn_frame, ROI_images


def update_periodical_data(frame_metrics: dict, periodical_data: dict, config: dict) -> dict:
    num_frames_new_head_nod = config["num_frames_new_head_nod"]
    num_frames_new_yawn = config["num_frames_new_yawn"]
    eye_alpha_val = config["eye_alpha_val"]

    blink_value = 0
    head_nod_value = 0
    yawn_value = 0
    eye_state_value = 0

    periodical_data["frame_count"] += 1
    if frame_metrics is None:
        return periodical_data

    periodical_data["current_head_nod_state"] = frame_metrics["head_nod"]
    if frame_metrics["head_nod"]:
        periodical_data["head_nod_total_frame_count"] += 1

        # si detectamos un head nod y en el frame anterior no lo hicimos
        if (not periodical_data["previous_head_nod_state"] and
            periodical_data["frame_count"] - periodical_data["head_nod_last_frame"] > num_frames_new_head_nod):
            periodical_data["head_nod_count"] += 1
            head_nod_value = 1
    else:
        if periodical_data["previous_head_nod_state"]:
            periodical_data["head_nod_last_frame"] = periodical_data["frame_count"] - 1


    periodical_data["current_yawn_state"] = frame_metrics["yawn"]
    if frame_metrics["yawn"]:
        periodical_data["yawn_total_frame_count"] += 1

        if (not periodical_data["previous_yawn_state"]):
            # periodical_data["frame_count"] - periodical_data["yawn_last_frame"] > num_frames_new_yawn):
            periodical_data["num_yawns"] += 1
            yawn_value = 1
    else:
        if periodical_data["previous_yawn_state"]:
            periodical_data["yawn_last_frame"] = periodical_data["frame_count"] - 1


    if frame_metrics["open_eyes"]:
        periodical_data["current_eye_state"] = "open"
        
        if periodical_data["current_frames_closed_eyes"] > periodical_data["max_frames_closed_eyes"]:
            periodical_data["max_frames_closed_eyes"] = periodical_data["current_frames_closed_eyes"]
        
        if periodical_data["current_frames_closed_eyes"] > 0:
            periodical_data["num_blinks"] += 1
            blink_value = 1
        # #### check for "open closed open" case
        # if periodical_data["previous_frame_eye_state"] == "closed" and periodical_data["previous2_frame_eye_state"] == "open":
        #     periodical_data["previous_frame_eye_state"] = "open"
        #     periodical_data["num_blinks"] -= 1
        #     periodical_data["closed_eye_frame_count"] -= 1

        periodical_data["previous_current_frames_closed_eyes"] = periodical_data["current_frames_closed_eyes"]
        periodical_data["current_frames_closed_eyes"] = 0
    else:
        periodical_data["current_eye_state"] = "closed"
        eye_state_value = 1

        # #### check for "closed open closed" case
        # if periodical_data["previous_frame_eye_state"] == "open" and periodical_data["previous2_frame_eye_state"] == "closed":
        #     periodical_data["previous_frame_eye_state"] = "closed"
        #     periodical_data["closed_eye_frame_count"] += 1
        #     periodical_data["current_frames_closed_eyes"] = periodical_data["previous_current_frames_closed_eyes"] + 1
        # elif periodical_data["previous_frame_eye_state"] == "open":
        #periodical_data["num_blinks"] += 1
        
        periodical_data["closed_eye_frame_count"] += 1
        periodical_data["current_frames_closed_eyes"] += 1

    periodical_data["previous_yawn_state"] = periodical_data["current_yawn_state"]
    periodical_data["previous_head_nod_state"] = periodical_data["current_head_nod_state"]
    periodical_data["previous2_frame_eye_state"] = periodical_data["previous_frame_eye_state"]
    periodical_data["previous_frame_eye_state"] = periodical_data["current_eye_state"]
    #periodical_data["ear_values"].append(frame_metrics["ear"])
    periodical_data["sum_ear"] += frame_metrics["ear"]

    if frame_metrics["open_eyes"]:
        periodical_data["expected_ear"] = periodical_data["expected_ear"] * (1 - eye_alpha_val) + frame_metrics["ear"] * eye_alpha_val

    periodical_data_max_values = config["periodical_data_max_values"]
    if len(periodical_data["blink_values"]) >= periodical_data_max_values:
        periodical_data["blink_values"].pop(0)
    periodical_data["blink_values"].append(blink_value)
    
    if len(periodical_data["yawn_values"]) >= periodical_data_max_values:
        periodical_data["yawn_values"].pop(0)
    periodical_data["yawn_values"].append(yawn_value)
    
    if len(periodical_data["head_nod_values"]) >= periodical_data_max_values:
        periodical_data["head_nod_values"].pop(0)
    periodical_data["head_nod_values"].append(head_nod_value)
    
    if len(periodical_data["closed_eye_frame_values"]) >= periodical_data_max_values:
        periodical_data["closed_eye_frame_values"].pop(0)
    periodical_data["closed_eye_frame_values"].append(eye_state_value)


    if len(periodical_data["nose_tip_y_values"]) >= config["num_frames_nose_tip_y"]:
        periodical_data["nose_tip_y_values"].pop(0)
    periodical_data["nose_tip_y_values"].append(frame_metrics["nose_tip_y"])

    # if len(periodical_data["mouth_top_y_values"]) >= config["num_frames_mouth_top_y"]:
    #     periodical_data["mouth_top_y_values"].pop(0)
    # periodical_data["mouth_top_y_values"].append(frame_metrics["mouth_top_y"])


    if len(periodical_data["ear_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["ear_values"].pop(0)
    periodical_data["ear_values"].append(frame_metrics["ear"])

    if frame_metrics["open_eyes"]:
        if len(periodical_data["open_ear_values"]) >= config["num_ear_past_frames"]:
            periodical_data["open_ear_values"].pop(0)
        periodical_data["open_ear_values"].append(frame_metrics["ear"])

        # if len(periodical_data["left_iris_prop"]) >= config["num_frames_dynamic_avg"]:
        #     periodical_data["left_iris_prop"].pop(0)
        # periodical_data["left_iris_prop"].append(frame_metrics["left_iris_diameter"])

        # if len(periodical_data["right_iris_prop"]) >= config["num_frames_dynamic_avg"]:
        #     periodical_data["right_iris_prop"].pop(0)
        # periodical_data["right_iris_prop"].append(frame_metrics["right_iris_diameter"])

        # if len(periodical_data["left_euclidean_iris_prop"]) >= config["num_frames_dynamic_avg"]:
        #     periodical_data["left_euclidean_iris_prop"].pop(0)
        # periodical_data["left_euclidean_iris_prop"].append(frame_metrics["left_euclidean_iris_diameter"])

        # if len(periodical_data["right_euclidean_iris_prop"]) >= config["num_frames_dynamic_avg"]:
        #     periodical_data["right_euclidean_iris_prop"].pop(0)
        # periodical_data["right_euclidean_iris_prop"].append(frame_metrics["right_euclidean_iris_diameter"])

    if len(periodical_data["mar_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["mar_values"].pop(0)
    periodical_data["mar_values"].append(frame_metrics["mar"])

    if len(periodical_data["pitch_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["pitch_values"].pop(0)
    periodical_data["pitch_values"].append(frame_metrics["pitch"])

    if len(periodical_data["yaw_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["yaw_values"].pop(0)
    periodical_data["yaw_values"].append(frame_metrics["yaw"])
    
    # if (periodical_data["frame_count"] - 1) % config["pose_angles_detection_frequency"] == 0:
    #     periodical_data["pose_angle_dict"] = {"yaw": frame_metrics["yaw"], "pitch": frame_metrics["pitch"], "roll": frame_metrics["roll"]} 

    if periodical_data["frame_count"] <= 20:
        # periodical_data["sum_first_mouth_width"] += frame_metrics["mouth_width"]
        periodical_data["mean_first_mouth_width"] = periodical_data["sum_first_mouth_width"] / min(20, periodical_data["frame_count"]) 
        # periodical_data["sum_left_iris_diameter"] += frame_metrics["left_iris_diameter"]
        # periodical_data["sum_right_iris_diameter"] += frame_metrics["right_iris_diameter"]
        # periodical_data["mean_left_iris_diameter"] = periodical_data["sum_left_iris_diameter"] / min(10, periodical_data["frame_count"])
        # periodical_data["mean_right_iris_diameter"] = periodical_data["sum_right_iris_diameter"] / min(10, periodical_data["frame_count"])
        periodical_data["sum_first_ear"] += frame_metrics["ear"]
        periodical_data["mean_first_ear"] = periodical_data["sum_first_ear"] / min(20, periodical_data["frame_count"])
        periodical_data["sum_first_pitch"] += frame_metrics["pitch"]
        periodical_data["mean_first_pitch"] = periodical_data["sum_first_pitch"] / min(20, periodical_data["frame_count"])


    return periodical_data

def compute_global_metrics(frame_metrics: dict, periodical_data: dict, config: dict, fps: int, frames_per_minute: int) -> dict:
    global_metrics = {}
    
    # global_metrics["mean_ear"] = periodical_data["sum_ear"] / periodical_data["frame_count"]
    global_metrics["blink_frequency"] = periodical_data["num_blinks"] / periodical_data["frame_count"]
    # global_metrics["blinks_per_minute"] = periodical_data["num_blinks"] * frames_per_minute / periodical_data["frame_count"]
    # global_metrics["perclos"] = periodical_data["closed_eye_frame_count"] / periodical_data["frame_count"]
    global_metrics["current_time_closed_eyes"] = periodical_data["current_frames_closed_eyes"] / fps
    # global_metrics["yawns_per_minute"] = periodical_data["num_yawns"] * frames_per_minute / periodical_data["frame_count"]
    # global_metrics["head_nods_per_minute"] = periodical_data["head_nod_count"] * frames_per_minute / periodical_data["frame_count"]
    global_metrics["mean_nose_tip_y"] = sum(periodical_data["nose_tip_y_values"]) / len(periodical_data["nose_tip_y_values"])
    # global_metrics["mean_mouth_top_y"] = sum(periodical_data["mouth_top_y_values"]) / len(periodical_data["mouth_top_y_values"])

    # if periodical_data["num_blinks"] == 0:
    #     global_metrics["mean_blink_time"] = 0
    # else:
    #     global_metrics["mean_blink_time"] = (periodical_data["closed_eye_frame_count"] / periodical_data["num_blinks"]) / fps

    ###########################
    # METODO VENTANA ULTIMO MIN

    values_length = len(periodical_data["closed_eye_frame_values"])
    # ratio = frames_per_minute / min(periodical_data["frame_count"], values_length)
    ratio = config["periodical_data_max_values"] / min(periodical_data["frame_count"], config["periodical_data_max_values"])
    num_blinks = sum(periodical_data["blink_values"])

    global_metrics["mean_ear"] = sum(periodical_data["ear_values"][:-values_length])/values_length
    global_metrics["blinks_per_minute"] = num_blinks * ratio
    global_metrics["perclos"] = sum(periodical_data["closed_eye_frame_values"])/values_length
    global_metrics["yawns_per_minute"] = sum(periodical_data["yawn_values"]) * ratio
    global_metrics["head_nods_per_minute"] = sum(periodical_data["head_nod_values"]) * ratio
    
    if num_blinks == 0:
        global_metrics["mean_blink_time"] = 0
    else:
        global_metrics["mean_blink_time"] = (sum(periodical_data["closed_eye_frame_values"]) / num_blinks) / fps

    return global_metrics


def obtain_frame_metrics(frame, periodical_data, config, obtain_global_metrics=False, fps=0):

    metrics = {}
    iris_diameters = None
    mean_first_ear = None
    mean_first_pitch = None
    
    detect_eyes = config["detect_eyes"]
    detect_mouth = config["detect_mouth"]
    detect_head = config["detect_head"]
    
    expected_ear = periodical_data["expected_ear"]

    last_iris_props = None
    if periodical_data["left_iris_prop"] != []:
        last_iris_props = {"left": periodical_data["left_iris_prop"], "right": periodical_data["right_iris_prop"],
                        "left_euclidean": periodical_data["left_euclidean_iris_prop"], 
                        "right_euclidean": periodical_data["right_euclidean_iris_prop"]}

    if "mean_first_ear" in periodical_data:
        mean_first_ear = periodical_data["mean_first_ear"]

    if "mean_first_pitch" in periodical_data:
        mean_first_pitch = periodical_data["mean_first_pitch"]

    last_pose_angles = None
    # TODO if periodical_data["frame_count"] % config["pose_angles_detection_frequency"] != 0:
    #     last_pose_angles = periodical_data["pose_angle_dict"]

    ear_values = None
    if periodical_data["open_ear_values"] != []:
        ear_values = periodical_data["open_ear_values"]  

    mar_values = None
    if periodical_data["mar_values"] != []:
        mar_values = periodical_data["mar_values"]  

    pitch_values = None
    if periodical_data["pitch_values"] != []:
        pitch_values = periodical_data["pitch_values"]    

    yaw_values = None
    if periodical_data["yaw_values"] != []:
        yaw_values = periodical_data["yaw_values"]  
    
    if periodical_data["yaw_values"] != []:
        yaw_values = periodical_data["yaw_values"]    

    nose_tip_y_values = None
    if periodical_data["nose_tip_y_values"] != []:
        nose_tip_y_values = periodical_data["nose_tip_y_values"]
    
    mouth_top_y_values = None
    if periodical_data["mouth_top_y_values"] != []:
        mouth_top_y_values = periodical_data["mouth_top_y_values"]

    previous_frame_eye_state = None
    dict = {"open": True, "closed": False}
    if periodical_data["previous_frame_eye_state"] is not None:
        previous_frame_eye_state = dict[periodical_data["previous_frame_eye_state"]]

    previous_frame_yawn_state = None
    if periodical_data["previous_yawn_state"] is not None:
        previous_frame_yawn_state = periodical_data["previous_yawn_state"]

    frame_metrics, drawn_frame, _ = process_frame(frame, config, eyes=detect_eyes, mouth=detect_mouth, head=detect_head, expected_ear=expected_ear, last_pose_angles=last_pose_angles, last_iris_prop_values=last_iris_props, framecount=periodical_data["frame_count"], mean_first_ear=mean_first_ear, mean_first_pitch=mean_first_pitch, last_pitch_values=pitch_values, last_yaw_values=yaw_values, last_mar_values=mar_values, last_ear_values=ear_values, nose_tip_y_values=nose_tip_y_values, mouth_top_y_values=mouth_top_y_values, previous_eye_state=previous_frame_eye_state, previous_yawn_state=previous_frame_yawn_state)
    metrics["frame_metrics"] = frame_metrics
    # TODO: periodical data que tenga en cuenta info de los ultimos x minutos
    periodical_data = update_periodical_data(frame_metrics, periodical_data, config)
    metrics["periodical_data"] = periodical_data
    metrics["drawn_frame"] = drawn_frame

    if frame_metrics is not None:
        if obtain_global_metrics:
            global_metrics = compute_global_metrics(frame_metrics, periodical_data, config, fps, int(60 * fps))
            global_metrics["frame"] = periodical_data["frame_count"] - 1
            metrics["global_metrics"] = global_metrics

    return metrics



def obtencion_metricas_locales_frame(row, index_dict, periodical_data, config):
    ###### EYE INFO ########
    eye_method = config["eye_closure_method"]
    eye_threshold = config["eye_closure_threshold"]
    eye_threshold_perc = config["main_threshold_perc"]
    eye_gray_zone_perc = config["gray_zone_perc"]
    num_ear_past_frames = config["num_ear_past_frames"]
    eye_alpha_val = config["eye_alpha_val"]

    expected_ear = periodical_data["expected_ear"]
    mean_first_ear = periodical_data["mean_first_ear"]
    last_ear_values = periodical_data["open_ear_values"]
    eye_state_dict = {"open": True, "closed": False, None: None}
    previous_eye_state = eye_state_dict[periodical_data["previous_frame_eye_state"]]
    ########################
    ###### MOUTH INFO ######
    mouth_method = config["mouth_closure_method"]
    yawn_threshold = config["mouth_yawn_threshold"]
    yawn_gray_zone = config["gray_zone_mouth"]
    num_past_frames_mouth = config["num_past_frames_mouth"]

    last_mar_values = periodical_data["mar_values"]
    previous_mouth_state = periodical_data["previous_yawn_state"]
    ########################
    ###### HEAD INFO #######
    head_method = config["pitch_method"]
    head_nod_threshold = config["head_nod_threshold"]
    head_nod_threshold_perc = config["head_nod_threshold_perc"]
    nose_tip_y_diff_threshold = config["head_nod_y_min_threshold"]
    num_past_frames_pitch = config["num_past_frames_pitch"]
    gray_zone_pitch = config["gray_zone_pitch"]
    previous_head_nod_state = periodical_data["previous_head_nod_state"]

    mean_first_pitch = periodical_data["mean_first_pitch"]
    last_pitch_values = periodical_data["pitch_values"]
    nose_tip_y_values = periodical_data["nose_tip_y_values"]
    ########################

    frame_metrics = {}

    # if mean_first_ear is not None:
    #     eye_threshold = mean_first_ear * eye_threshold_perc

    # if last_ear_values != [] and num_ear_past_frames > 0:
    #     ear_last = min(num_ear_past_frames, len(last_mar_values))
    #     ear_values_to_analyze = last_ear_values[-ear_last:]
    #     eye_threshold = eye_threshold_perc * sum(ear_values_to_analyze) / len(ear_values_to_analyze)

    eye_gray_zone = 0
    if expected_ear != 0:
        eye_threshold = eye_threshold_perc * expected_ear
        eye_gray_zone = eye_gray_zone_perc * expected_ear

    ear = row[index_dict[f"ear{eye_method}"]]
    mean_ear = ear

    eye_state = image_analysis.check_eyes_open(mean_ear, eye_threshold, eye_gray_zone, previous_eye_state)
    frame_metrics["open_eyes"] = eye_state
    frame_metrics["ear"] = ear

    mar = row[index_dict[f"mar{mouth_method}"]]
    mean_mar = mar
    if last_mar_values != [] and num_past_frames_mouth > 0:
        mar_last = min(num_past_frames_mouth, len(last_mar_values))
        mar_values_to_analyze = last_mar_values[-mar_last:]
        mean_mar = (sum(mar_values_to_analyze) + mar) / (mar_last + 1)
    
    yawn = image_analysis.check_yawn(mean_mar, yawn_threshold, yawn_gray_zone, previous_mouth_state) 
    frame_metrics["yawn"] = yawn
    frame_metrics["mar"] = mar

    # if mean_first_pitch is not None:
    #     head_nod_threshold = mean_first_pitch * head_nod_threshold_perc

    pitch = row[index_dict[f"pitch{head_method}"]]
    mean_pitch = pitch
    if last_pitch_values != [] and num_past_frames_pitch > 0:
        pitch_last = min(num_past_frames_pitch, len(last_pitch_values))
        pitch_values_to_analyze = last_pitch_values[-pitch_last:]
        mean_pitch = (sum(pitch_values_to_analyze) + pitch) / (pitch_last + 1)
    
    if last_pitch_values != []:
        head_nod_threshold = head_nod_threshold_perc * sum(last_pitch_values) / len(last_pitch_values)

    nose_tip_y = row[index_dict["nose_tip_y"]]
    possible_head_nod = True #not eye_state
    # if possible_head_nod and nose_tip_y_values != []:
    #     mean_nose_tip_y = sum(nose_tip_y_values) / len(nose_tip_y_values)
    #     nose_tip_y_threshold = mean_nose_tip_y + nose_tip_y_diff_threshold
    #     possible_head_nod = nose_tip_y >= nose_tip_y_threshold

    head_nod = False
    if possible_head_nod:
        # head_nod = image_analysis.check_head_nod(mean_pitch, head_nod_threshold)
        upper_threshold = head_nod_threshold * (1 + gray_zone_pitch)
        lower_threshold = head_nod_threshold * (1 - gray_zone_pitch)
        if mean_pitch < lower_threshold:
            head_nod = True
        elif mean_pitch > upper_threshold:
            head_nod = False
        else:
            head_nod = mean_pitch <= head_nod_threshold
            if previous_head_nod_state is not None:
                head_nod = previous_head_nod_state
    
    frame_metrics["head_nod"] = head_nod
    frame_metrics["pitch"] = pitch
    frame_metrics["nose_tip_y"] = nose_tip_y

    yaw = row[index_dict[f"yaw"]]
    # mean_yaw = yaw
    # if last_yaw_values != []:
    #     yaw_last = min(2, len(last_yaw_values))
    #     yaw_values_to_analyze = last_yaw_values[-yaw_last:]
    #     mean_yaw = (sum(yaw_values_to_analyze) + yaw) / (yaw_last + 1)
    frame_metrics["yaw"] = yaw
    return frame_metrics


def obtain_metrics_from_df(df, config):
    '''
        Este metodo tiene como objetivo simular la obtencion de metricas que se realiza sobre los videos,
        pero utilizando dataframes como target.

        Estructura:
        por cada fila:
            computar metricas del frame (open eyes, yawn, headnod)
            computar metricas temporales (periodical_data, global_metrics)
        
        Salida:
            Un nuevo csv, que contenga las metricas temporales
    '''
    index_dict = { name: i for i, name in enumerate(list(df), start=0) }

    periodical_data = copy.deepcopy(periodical_data_initial_state)
    metrics_list = []
    labels = ["drowsiness", "mouth", "head", "eye"]
    available_labels = [ label for label in labels if label in index_dict ]
    metrics_of_interest = config["metrics_to_obtain"] + labels
    ind = 0
    for row in df.itertuples(name=None, index=False):
        fps = row[index_dict["fps"]]
    
        if math.isnan(row[index_dict["frame_count"]]):
            # si no hemos podido detectar la cara, skipeamos el frame
            # print(f"Skipping frame {ind}...")
            periodical_data["frame_count"] += 1
            flat_metrics = [ (label, row[index_dict[label]]) for label in available_labels ]
            selected_metrics = { metric:value for metric, value in flat_metrics if metric in metrics_of_interest }
            metrics_list.append(selected_metrics)
            ind += 1
            continue

        frame_metrics = obtencion_metricas_locales_frame(row, index_dict, periodical_data, config)
        periodical_data = update_periodical_data(frame_metrics, periodical_data, config)
        global_metrics = compute_global_metrics(frame_metrics, periodical_data, config, int(fps), int(fps * 60))
        
        flat_metrics = dict(list(global_metrics.items())  
                          + list(periodical_data.items())
                          + list(frame_metrics.items())
                          + [ (label, row[index_dict[label]]) for label in available_labels ])
        selected_metrics = { metric:value for metric, value in flat_metrics.items() if metric in metrics_of_interest }
        metrics_list.append(selected_metrics)
        ind += 1
    
    resulting_df = pd.DataFrame(metrics_list)
    return resulting_df


def obtain_metrics_from_video(input_video, subject, config):
    periodical_data = copy.deepcopy(periodical_data_initial_state)

    metrics = []
    remaining_frames_of_period = config["period_length"]
    fps = round(input_video.get(cv2.CAP_PROP_FPS))
    start = time.time()
    valid_frame, frame = input_video.read()
    while valid_frame: # and periodical_data["frame_count"] < 1000:
        remaining_frames_of_period -= 1
        obtain_global_metrics = False
        if remaining_frames_of_period <= 0:
            obtain_global_metrics = True
            remaining_frames_of_period = config["period_length"]

        current_metrics = obtain_frame_metrics(frame, periodical_data, config, obtain_global_metrics, fps)
        periodical_data = current_metrics["periodical_data"]

        if obtain_global_metrics:
            if current_metrics["frame_metrics"] is not None:
                flat_metrics = dict(list(current_metrics["global_metrics"].items()) 
                                   + list(current_metrics["frame_metrics"].items()) 
                                   + list(current_metrics["periodical_data"].items()))
                selected_metrics = { metric:value for metric, value in flat_metrics.items() if metric in config["metrics_to_obtain"] }

                metrics.append(selected_metrics)
            else:
                selected_metrics = {metric:None for metric in config["metrics_to_obtain"]}
                metrics.append(selected_metrics)


        valid_frame, frame = input_video.read()
        if periodical_data["frame_count"] % 1000 == 0:
            print(f"{periodical_data['frame_count']}: {time.time() - start}")

    return metrics


def create_dataset_from_video(input_video, subject, config, labels):

    metric_list = obtain_metrics_from_video(input_video, subject, config)
    metric_dataframe = pd.DataFrame(metric_list)

    fps = round(input_video.get(cv2.CAP_PROP_FPS))
    num_samples = len(metric_list)

    for label_type, values in labels.items():
        metric_dataframe[label_type] = values
    metric_dataframe["fps"] = [fps] * num_samples
    metric_dataframe["subject"] = [subject] * num_samples
    return metric_dataframe


def create_dataset_from_videos_NTHU(videos_and_labels: dict, target_folder: str, config: dict) -> list:
    df_dataset_dict = {}
    for dataset, dataset_dict in videos_and_labels.items():
        dataset_target_folder = os.path.join(target_folder, dataset)
        df_list = []
        subjects_to_analyze = config["subjects"][dataset]
        scenarios_to_analyze = config["scenarios"]
        states_to_analyze = config["states"]

        subjects = { subject:data for subject, data in dataset_dict.items() if subject in subjects_to_analyze }
        for subject, scenarios in subjects.items():
            scenarios = { scenario:data for scenario, data in scenarios.items() if scenario in scenarios_to_analyze }
            for scenario, driver_states in scenarios.items():
                driver_states = { state:data for state, data in driver_states.items() if state in states_to_analyze and data != {} }
                for driver_state, vid_lab in driver_states.items():
                    print(vid_lab)
                    labels = vid_lab["labels"]
                    video = vid_lab["video"]
                    print(f"{subject}_{scenario}_{driver_state}")
                    video_df = create_dataset_from_video(video, subject, config, labels)
                    target_path = os.path.join(dataset_target_folder, f"{subject}_{scenario}_{driver_state}.csv")
                    video_df.to_csv(target_path)
                    df_list.append(video_df)
        
        df_dataset_dict[dataset] = df_list

    return df_dataset_dict

def load_source_dfs_NTHU(source_folder):
    source_dfs = { "train": {}, "test": {} }

    for dataset in ["train", "test"]:
        folder = os.path.join(source_folder, dataset)
        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)
            if filename[-4:] == ".csv" and "big_df" not in filename:
                df = pd.read_csv(file) 
                df = df.drop("Unnamed: 0", axis=1)
                source_dfs[dataset][filename] = df 
    
    return source_dfs

def create_dataset_from_df_NTHU(source_folder: str, target_folder: str, config: dict) -> list:
    source_dfs = load_source_dfs_NTHU(source_folder)
    
    df_dataset_dict = {}
    big_df_list = []
    for dataset, dataframe_dict in source_dfs.items():
        dataset_target_folder = os.path.join(target_folder, dataset)
        df_list = []

        for name, df in dataframe_dict.items():
            print(name)
            temporal_metrics_df = obtain_metrics_from_df(df, config)
            temporal_metrics_df["dataset"] = dataset
            target_path = os.path.join(dataset_target_folder, name)
            temporal_metrics_df.to_csv(target_path)
            df_list.append(temporal_metrics_df)
            big_df_list.append(temporal_metrics_df)
        
        df_dataset_dict[dataset] = df_list
        big_dataset_df = pd.concat(df_list)
        big_dataset_df.to_csv(os.path.join(dataset_target_folder, "big_df.csv"))

    big_df = pd.concat(big_df_list)
    big_df.to_csv(os.path.join(target_folder, "big_df.csv"))
    return df_dataset_dict

def create_dataset_from_videos_eyeblink8(videos_and_labels: dict, target_folder: str, config: dict) -> list:
    df_list = []
    # subjects_to_analyze = config["subjects"]
    # scenarios_to_analyze = config["scenarios"]
    # states_to_analyze = config["states"]

    # subjects = { subject:data for subject, data in dataset_dict.items() if subject in subjects_to_analyze }
    for subject, data in videos_and_labels.items():
        video = data["video"]
        metric_list = obtain_metrics_from_video(video, subject, config)
        metric_df = pd.DataFrame(metric_list)[["ear1", "ear2", "ear3, ear4"]]
        metric_df["subject"] = subject
        target_path = os.path.join(target_folder, f"{subject}.csv")
        metric_df.to_csv(target_path)
        df_list.append(metric_df)

    df = pd.concat(df_list)
    return df


def create_dataset_from_images_CEW(images, labels, target_folder: str, config: dict) -> list:
    metric_list = []
    ind = 0
    for image in images:
        result, _, _ = process_frame(image, config, mouth=False, head=False)
        metrics = {}
        if result is not None:
            for ear_type in ["ear1", "ear2", "ear3, ear4"]:
                metrics[ear_type] = result[ear_type]
        
        metrics["label"] = labels[ind]
        metric_list.append(metrics)
        ind += 1
    
    df = pd.DataFrame(metric_list)
    df.to_csv(f"{target_folder}big_df.csv")
    return df


def create_dataset_from_videos(path, target_folder, config) -> list:
    df_list = []
    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        video_extensions = [ ".mp4", ".mov", ".avi", ".mp3" ]

        if os.path.isdir(file):
            df_list = df_list + create_dataset_from_videos(file, target_folder, config)
        elif os.path.isfile(file) and filename[-4:].lower() in video_extensions:
            video = cv2.VideoCapture(file)
            print(file)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            file = file.replace('\\', '/')
            subject = file.split('/')[-2]
            print(subject)
            if filename[0] == "0":
                labels = [0] * frame_count
                df = create_dataset_from_video(video, subject, config, labels)
                df.to_csv(f"{target_folder}{subject}_{labels[0]}.csv")
                df_list.append(df)
            elif filename[0] == "1":
                labels = [10] * frame_count
                df = create_dataset_from_video(video, subject, config, labels)             
                suffix = ""
                if filename[-5] != "0":
                    suffix = f"_{filename[-5]}"

                df.to_csv(f"{target_folder}{subject}_{labels[0]}{suffix}.csv")
                df_list.append(df)

    return df_list