from lib2to3.pgen2 import driver
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
# import tensorflow as tf

import region_detection as roi
import image_analysis


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
                res_img = cv2.circle(res_img, point, radius=1, color=(0, 0, 255), thickness=-1)
                #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def draw_lip_landmarks(img, face_landmarks, lip_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for lip_pos, landmarks in lip_indexes.items():
        for ind in landmarks:
            point = face_landmarks[ind]
            point = (int(point.x*width), int(point.y*height))
            res_img = cv2.circle(res_img, point, radius=1, color=(0, 0, 255), thickness=-1)
            #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def draw_iris_landmarks(img, face_landmarks, iris_indexes: dict):
    res_img = img.copy()
    height, width, _ = res_img.shape
    for iris, indexes in iris_indexes.items():
        for ind in indexes:
            point = face_landmarks[ind]
            point = (int(point.x*width), int(point.y*height))
            res_img = cv2.circle(res_img, point, radius=1, color=(0, 0, 255), thickness=-1)
            #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.1)

    return res_img

def process_frame(frame, config, mean_iris_props=None, mean_first_ear=None, mean_first_pitch=None, last_pitch_values=None, last_yaw_values=None, last_mar_values=None, last_ear_values=None, nose_tip_y_values=None, mouth_top_y_values=None, previous_eye_state=None, framecount=None):
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

        # pose_angle_dict, drawn_frame = roi.estimate_head_pose(frame, drawn_frame, face_landmarks, debug=False)
        pose_angle_dict, drawn_frame = roi.estimate_head_pose_model(frame, (x, y, x1, y1), debug=False)

        ## FACE BOUNDING BOX ################################################################
        # if framecount is not None:
        #     print(f"FRAME: {framecount+1}")

        width = right - left
        height = bot - top
        face_dimensions = (width * frame_width, height * frame_height)
        indexes = (top_ind, bot_ind, left_ind, right_ind)
        #print([face_landmarks[ind] for ind in indexes])
        #print(f"width: {width}, height: {height}")
        #####################################################################################
        ## left
        top_left = face_landmarks[470].y
        bot_left = face_landmarks[472].y
        iris_diameters_left = (bot_left - top_left)
        iris_prop_left = iris_diameters_left / height

        ## right
        top_right = face_landmarks[475].y
        bot_right = face_landmarks[477].y
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

        left_iris_indexes = [ 468, 469, 470, 471, 472 ]
        right_iris_indexes = [ 473, 474, 475, 476, 477 ]
        iris_indexes = { "left": left_iris_indexes, "right": right_iris_indexes}

        # upper_lip_indexes = [ 81, 82, 13, 312, 311 ]
        # lower_lip_indexes = [ 178, 87, 14, 317, 402 ]
        upper_lip_indexes = [ 82, 13, 312 ]
        lower_lip_indexes = [ 87, 14, 317 ]
        center_lip_indexes = [ 78, 308 ]
        lip_indexes = { "upper_landmarks": upper_lip_indexes, "lower_landmarks": lower_lip_indexes, "center_landmarks": center_lip_indexes}
        
        nose_tip_y = face_landmarks[1].y
        mouth_top_y = (face_landmarks[17].y - face_landmarks[0].y) / height # dividimos entre la altura de la cabeza
        
        right_mouth = np.array([face_landmarks[308].x, face_landmarks[308].y])
        left_mouth = np.array([face_landmarks[78].x, face_landmarks[78].y])
        mouth_width = (face_landmarks[308].x - face_landmarks[78].x) / width
        # mouth_width = np.linalg.norm(right_mouth - left_mouth)
        # cv2.imshow("", res_img)
        #cv2.waitKey()

        ROI_images = roi.get_ROI_images(frame, face_dimensions, face_landmarks)
        cv2.imwrite(f"output_eyes/eye{framecount}.jpg", ROI_images[f"{closer_eye}_eye"])

        #print(tensorflow_open_close(["eye.jpg"]))

        #cv2.imwrite("right_eye.jpg", ROI_images["right_eye"])

        #iris_centers = roi.get_iris_centers(frame, face_landmarks, iris_indexes)

        iris_metrics = roi.get_iris_metrics(frame, face_landmarks, iris_indexes)
        method = config["eye_closure_method"]
        if method == 1:
            eye_closure_func = image_analysis.compute_eye_closure1
        elif method == 2:
            eye_closure_func = image_analysis.compute_eye_closure2
        else:
            eye_closure_func = image_analysis.compute_eye_closure3

        nose_tip_y_diff_threshold = config["head_nod_y_min_threshold"]
        mouth_top_y_diff_threshold = config["yawn_y_min_threshold"]

        head_nod_threshold = config["head_nod_threshold"]
        head_nod_threshold_perc = config["head_nod_threshold_perc"]

        yawn_threshold = config["mouth_yawn_threshold"]

        threshold = config["eye_closure_threshold"]

        main_threshold_perc = config["main_threshold_perc"]
        gray_zone_perc = config["gray_zone_perc"]
        if mean_first_ear is not None:
            threshold = mean_first_ear * main_threshold_perc
        
        if last_ear_values is not None:
            threshold = main_threshold_perc * sum(last_ear_values) / len(last_ear_values)

        if mean_first_pitch is not None:
            head_nod_threshold = mean_first_pitch * head_nod_threshold_perc
        
        num_frames_lag_ignore = config["num_frames_lag_ignore"]
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
        
        mar = image_analysis.compute_mouth_closure(frame, face_landmarks, **lip_indexes)
        mean_mar = mar
        if last_mar_values is not None:
            mar_last = min(2, len(last_mar_values))
            mar_values_to_analyze = last_mar_values[-mar_last:]
            mean_mar = (sum(mar_values_to_analyze) + mar) / (mar_last + 1)
        
        ear = eye_closure_func(frame, face_landmarks, iris_diameter=iris_diameters[closer_eye], **eye_indexes[closer_eye])
        mean_ear = ear
        # if last_ear_values is not None:
        #     ear_last = min(2, len(last_ear_values))
        #     ear_values_to_analyze = last_ear_values[-ear_last:]
        #     mean_ear = (sum(ear_values_to_analyze) + ear) / (ear_last + 1)
        
        open_eyes = image_analysis.check_eyes_open(mean_ear, threshold, gray_zone_perc, previous_eye_state)

        possible_head_nod = not open_eyes
        if not open_eyes and nose_tip_y_values is not None:
            mean_nose_tip_y = sum(nose_tip_y_values) / len(nose_tip_y_values)
            nose_tip_y_threshold = mean_nose_tip_y + nose_tip_y_diff_threshold
            possible_head_nod = nose_tip_y >= nose_tip_y_threshold
        
        if mouth_top_y_values is not None:
            mean_mouth_top_y = sum(mouth_top_y_values) / len(mouth_top_y_values)
            mouth_top_y_threshold = mean_mouth_top_y + mouth_top_y_diff_threshold
            # possible_yawn = mouth_top_y >= mouth_top_y_threshold

        # if last_pitch_values is not None and framecount > num_frames_lag_ignore:
        #     pitch_values_to_analyze = last_pitch_values[:-num_frames_lag_ignore]
        #     head_nod_threshold = head_nod_threshold_perc * sum(pitch_values_to_analyze) / len(pitch_values_to_analyze)


        
        head_nod = possible_head_nod
        if possible_head_nod:
            head_nod = image_analysis.check_head_nod(mean_pitch, head_nod_threshold)
        
        yawn = image_analysis.check_yawn(mean_mar, yawn_threshold)

        frame_metrics = {}
        frame_metrics["mean_yaw_3_frames"] = mean_yaw
        frame_metrics["mean_pitch_3_frames"] = mean_pitch
        frame_metrics["mean_ear_3_frames"] = mean_ear
        frame_metrics["mean_mar_3_frames"] = mean_mar
        # frame_metrics["lear"] = eye_closure_func(frame, face_landmarks, iris_diameter=iris_diameters["left"], **eye_indexes["left"])
        # frame_metrics["rear"] = eye_closure_func(frame, face_landmarks, iris_diameter=iris_diameters["right"], **eye_indexes["right"])
        frame_metrics["ear"] = ear
        frame_metrics["open_eyes"] = open_eyes
        frame_metrics["yawn"] = yawn
        frame_metrics["mar"] = mar
        frame_metrics["left_iris_diameter"] = iris_prop_left
        frame_metrics["right_iris_diameter"] = iris_prop_right
        frame_metrics["yaw"] = pose_angle_dict["yaw"]
        frame_metrics["pitch"] = pose_angle_dict["pitch"]
        frame_metrics["head_nod"] = head_nod
        frame_metrics["nose_tip_y"] = nose_tip_y
        frame_metrics["mouth_top_y"] = mouth_top_y
        frame_metrics["mouth_width"] = mouth_width

        #indexes = [27, 28, 29, 22, 23, 24, 257, 258, 259, 252, 253, 254]
        drawn_frame = draw_lip_landmarks(drawn_frame, face_landmarks, lip_indexes)
        drawn_frame = draw_eye_landmarks(drawn_frame, face_landmarks, {"eye": closer_eye_indexes})
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
    else:
        if periodical_data["previous_head_nod_state"]:
            periodical_data["head_nod_last_frame"] = periodical_data["frame_count"] - 1


    periodical_data["current_yawn_state"] = frame_metrics["yawn"]
    if frame_metrics["yawn"]:
        periodical_data["yawn_total_frame_count"] += 1

        if (not periodical_data["previous_yawn_state"] and
            periodical_data["frame_count"] - periodical_data["yawn_last_frame"] > num_frames_new_yawn):
            periodical_data["num_yawns"] += 1
    else:
        if periodical_data["previous_yawn_state"]:
            periodical_data["yawn_last_frame"] = periodical_data["frame_count"] - 1


    if frame_metrics["open_eyes"]:
        periodical_data["current_eye_state"] = "open"
        
        if periodical_data["current_frames_closed_eyes"] > periodical_data["max_frames_closed_eyes"]:
            periodical_data["max_frames_closed_eyes"] = periodical_data["current_frames_closed_eyes"]
        
        if periodical_data["current_frames_closed_eyes"] > 0:
            periodical_data["num_blinks"] += 1
        # #### check for "open closed open" case
        # if periodical_data["previous_frame_eye_state"] == "closed" and periodical_data["previous2_frame_eye_state"] == "open":
        #     periodical_data["previous_frame_eye_state"] = "open"
        #     periodical_data["num_blinks"] -= 1
        #     periodical_data["closed_eye_frame_count"] -= 1

        periodical_data["previous_current_frames_closed_eyes"] = periodical_data["current_frames_closed_eyes"]
        periodical_data["current_frames_closed_eyes"] = 0
    else:
        periodical_data["current_eye_state"] = "closed"

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

    if len(periodical_data["nose_tip_y_values"]) >= config["num_frames_nose_tip_y"]:
        periodical_data["nose_tip_y_values"].pop(0)
    periodical_data["nose_tip_y_values"].append(frame_metrics["nose_tip_y"])

    if len(periodical_data["mouth_top_y_values"]) >= config["num_frames_mouth_top_y"]:
        periodical_data["mouth_top_y_values"].pop(0)
    periodical_data["mouth_top_y_values"].append(frame_metrics["mouth_top_y"])


    if len(periodical_data["ear_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["ear_values"].pop(0)
    periodical_data["ear_values"].append(frame_metrics["ear"])

    if len(periodical_data["mar_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["mar_values"].pop(0)
    periodical_data["mar_values"].append(frame_metrics["mar"])

    if len(periodical_data["pitch_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["pitch_values"].pop(0)
    periodical_data["pitch_values"].append(frame_metrics["pitch"])

    if len(periodical_data["yaw_values"]) >= config["num_frames_dynamic_avg"]:
        periodical_data["yaw_values"].pop(0)
    periodical_data["yaw_values"].append(frame_metrics["yaw"])


    if periodical_data["frame_count"] <= 20:
        periodical_data["sum_first_mouth_width"] += frame_metrics["mouth_width"]
        periodical_data["mean_first_mouth_width"] = periodical_data["sum_first_mouth_width"] / min(20, periodical_data["frame_count"]) 
        periodical_data["sum_left_iris_diameter"] += frame_metrics["left_iris_diameter"]
        periodical_data["sum_right_iris_diameter"] += frame_metrics["right_iris_diameter"]
        periodical_data["mean_left_iris_diameter"] = periodical_data["sum_left_iris_diameter"] / min(10, periodical_data["frame_count"])
        periodical_data["mean_right_iris_diameter"] = periodical_data["sum_right_iris_diameter"] / min(10, periodical_data["frame_count"])
        periodical_data["sum_first_ear"] += frame_metrics["ear"]
        periodical_data["mean_first_ear"] = periodical_data["sum_first_ear"] / min(20, periodical_data["frame_count"])
        periodical_data["sum_first_pitch"] += frame_metrics["pitch"]
        periodical_data["mean_first_pitch"] = periodical_data["sum_first_pitch"] / min(20, periodical_data["frame_count"])


    return periodical_data

def compute_global_metrics(frame_metrics: dict, periodical_data: dict, fps: int, frames_per_minute: int) -> dict:
    global_metrics = {}
    
    global_metrics["mean_ear"] = periodical_data["sum_ear"] / periodical_data["frame_count"]
    global_metrics["blink_frequency"] = periodical_data["num_blinks"] / periodical_data["frame_count"]
    global_metrics["blinks_per_minute"] = periodical_data["num_blinks"] * frames_per_minute / periodical_data["frame_count"]
    global_metrics["perclos"] = periodical_data["closed_eye_frame_count"] / periodical_data["frame_count"]
    global_metrics["current_time_closed_eyes"] = periodical_data["current_frames_closed_eyes"] / fps
    global_metrics["yawns_per_minute"] = periodical_data["num_yawns"] * frames_per_minute / periodical_data["frame_count"]
    global_metrics["head_nods_per_minute"] = periodical_data["head_nod_count"] * frames_per_minute / periodical_data["frame_count"]
    global_metrics["mean_nose_tip_y"] = sum(periodical_data["nose_tip_y_values"]) / len(periodical_data["nose_tip_y_values"])
    global_metrics["mean_mouth_top_y"] = sum(periodical_data["mouth_top_y_values"]) / len(periodical_data["mouth_top_y_values"])

    if periodical_data["num_blinks"] == 0:
        global_metrics["mean_blink_time"] = 0
    else:
        global_metrics["mean_blink_time"] = (periodical_data["closed_eye_frame_count"] / periodical_data["num_blinks"]) / fps

    return global_metrics


def obtain_frame_metrics(frame, periodical_data, config, obtain_global_metrics=False, fps=0):

    metrics = {}
    iris_diameters = None
    mean_first_ear = None
    mean_first_pitch = None

    if "mean_left_iris_diameter" in periodical_data:
        iris_diameters = {"left": periodical_data["mean_left_iris_diameter"], "right": periodical_data["mean_right_iris_diameter"]}

    if "mean_first_ear" in periodical_data:
        mean_first_ear = periodical_data["mean_first_ear"]

    if "mean_first_pitch" in periodical_data:
        mean_first_pitch = periodical_data["mean_first_pitch"]


    ear_values = None
    if periodical_data["ear_values"] != []:
        ear_values = periodical_data["ear_values"]  

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

    frame_metrics, drawn_frame, _ = process_frame(frame, config, iris_diameters, framecount=periodical_data["frame_count"], mean_first_ear=mean_first_ear, mean_first_pitch=mean_first_pitch, last_pitch_values=pitch_values, last_yaw_values=yaw_values, last_mar_values=mar_values, last_ear_values=ear_values, nose_tip_y_values=nose_tip_y_values, mouth_top_y_values=mouth_top_y_values, previous_eye_state=previous_frame_eye_state)
    metrics["frame_metrics"] = frame_metrics
    # TODO: periodical data que tenga en cuenta info de los ultimos x minutos
    periodical_data = update_periodical_data(frame_metrics, periodical_data, config)
    metrics["periodical_data"] = periodical_data
    metrics["drawn_frame"] = drawn_frame

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
                    "previous_current_frames_closed_eyes": 0,
                    "max_frames_closed_eyes" : 0,
                    "mean_frames_closed_eyes" : 0,
                    "num_blinks" : 0,
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
                    "sum_first_pitch": 0,
                    "pitch_values": [],
                    "yaw_values": [],
                    "yawn_total_frame_count": 0,
                    "previous_yawn_state": False,
                    "yawn_last_frame": -1000,
                    "num_yawns" : 0,
                    "mar_values": [],
                    "nose_tip_y_values": [],
                    "mouth_top_y_values": [],
                    "sum_first_mouth_width": 0,
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
    df_list = []
    subjects_to_analyze = config["subjects"]
    scenarios_to_analyze = config["scenarios"]
    states_to_analyze = config["states"]

    print(videos_and_labels['001'].keys())
    print(subjects_to_analyze)

    subjects = { subject:data for subject, data in videos_and_labels.items() if subject in subjects_to_analyze }
    for subject, scenarios in subjects.items():
        scenarios = { scenario:data for scenario, data in scenarios.items() if scenario in scenarios_to_analyze }
        for scenario, driver_states in scenarios.items():
            driver_states = { state:data for state, data in driver_states.items() if state in states_to_analyze }
            for driver_state, vid_lab in driver_states.items():
                labels = vid_lab["labels"]
                video = vid_lab["video"]
                print(f"{subject}_{scenario}_{driver_state}")
                video_df = create_dataset_from_video(video, subject, config, labels)
                video_df.to_csv(f"{target_folder}{subject}_{scenario}_{driver_state}.csv")
                df_list.append(video_df)

    return df_list


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