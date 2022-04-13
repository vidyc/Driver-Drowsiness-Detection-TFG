import cv2
import random

import region_detection as roi
import image_analysis

input_type = "video"
max_num_frames = 5400

def process_frame(frame, config=None):
    annotated_frame, face_landmarks = roi.mediapipe_face_mesh(frame)

    if face_landmarks.multi_face_landmarks is None:
        return None

    left_eye_indexes = { "upper_landmarks": [158, 159], "lower_landmarks": [144, 145], "center_landmarks": [33, 133] }
    right_eye_indexes = { "upper_landmarks": [386, 385], "lower_landmarks": [374, 380], "center_landmarks": [263, 362] }
    eye_indexes = { "left_eye": left_eye_indexes, "right_eye": right_eye_indexes }

    res_img = frame.copy()
    height, width, _ = res_img.shape
    for eye, indexes in eye_indexes.items():
        for eye_pos, landmarks in indexes.items():
            for ind in landmarks:
                point = face_landmarks.multi_face_landmarks[0].landmark[ind]
                point = (int(point.x*width), int(point.y*height))
                res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
                res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)

    ROI_images = roi.get_ROI_images(frame, face_landmarks.multi_face_landmarks[0])
     
    open_eyes = image_analysis.check_eyes_open(frame, face_landmarks.multi_face_landmarks[0], eye_indexes)

    frame_metrics = {}
    # TODO: decidir si se computa la eye_closure como la media de los dos ojos
    frame_metrics["ear"] = image_analysis.compute_eye_closure(frame, face_landmarks.multi_face_landmarks[0], **eye_indexes["left_eye"])
    frame_metrics["open_eyes"] = open_eyes

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
    
    periodical_data["previous_frame_eye_state"] = periodical_data["current_eye_state"]
    periodical_data["ear_values"].append(frame_metrics["ear"])
    periodical_data["sum_ear"] += frame_metrics["ear"]

    return periodical_data

def compute_global_metrics(frame_metrics: dict, periodical_data: dict, frames_per_minute: int) -> dict:
    global_metrics = {}
    
    global_metrics["mean_ear"] = periodical_data["sum_ear"] / periodical_data["frame_count"]
    global_metrics["blink_frequency"] = periodical_data["num_blinks"] / periodical_data["frame_count"]
    global_metrics["blinks_per_minute"] = periodical_data["num_blinks"] * frames_per_minute / periodical_data["frame_count"]
    global_metrics["perclos"] = periodical_data["closed_eye_frame_count"] / periodical_data["frame_count"]

    return global_metrics

def compute_drowsiness_state(frame_metrics: dict, periodical_data: dict, global_metrics: dict, fps: int) -> dict:
    if global_metrics["perclos"] > 0.15 or global_metrics["blinks_per_minute"] < 10 \
       or periodical_data["current_frames_closed_eyes"] > fps * 0.5:
        return 10
        
    return 0


def inference_on_video(input_video):    
    periodical_data = { 
                        "frame_count" : 0,
                        "closed_eye_frame_count" : 0,
                        "current_frames_closed_eyes" : 0,
                        "max_frames_closed_eyes" : 0,
                        "mean_frames_closed_eyes" : 0,
                        "num_blinks" : 0,
                        "previous_frame_eye_state" : None,
                        "ear_values" : [], 
                        "sum_ear": 0,
                       }

    debug = False
    predictions = []
    valid_frame, frame = input_video.read()
    fps = input_video.get(cv2.CAP_PROP_FPS)
    print(fps)
    frames_per_minute = int(fps * 60)
    while valid_frame: # and periodical_data["frame_count"] < max_num_frames:
        frame_metrics = process_frame(frame)
        if frame_metrics is not None:
            periodical_data = update_periodical_data(frame_metrics, periodical_data)
            global_metrics = compute_global_metrics(frame_metrics, periodical_data, frames_per_minute)
            drowsiness_state = compute_drowsiness_state(frame_metrics, periodical_data, global_metrics, fps)
            predictions.append(drowsiness_state)
        
        valid_frame, frame = input_video.read()
        
        if periodical_data["frame_count"] % 1000 == 0:
            print(periodical_data["frame_count"])

        if debug:
            cv2.imshow('', frame)
            cv2.waitKey(0)

    
    print(frame_metrics)
    print()
    print(periodical_data)
    print()
    print(global_metrics)

    return predictions