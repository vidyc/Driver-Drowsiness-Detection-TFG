import cv2

import region_detection as roi
import image_analysis

input_type = "video"
max_num_frames = 20

def process_frame(frame, config=None):
    annotated_frame, face_landmarks = roi.mediapipe_face_mesh(frame)

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
    
    # TODO: determinar si los ojos estan abiertos o cerrados   
    open_eyes = image_analysis.check_eyes_open(frame, face_landmarks.multi_face_landmarks[0], eye_indexes)
    print(open_eyes)

    frame_metrics = {}
    # TODO: decidir si se computa la eye_closure como la media de los dos ojos
    frame_metrics["ear"] = image_analysis.compute_eye_closure(frame, face_landmarks.multi_face_landmarks[0], **eye_indexes["left_eye"])
    frame_metrics["open_eyes"] = open_eyes

    return frame_metrics


if input_type == "video":
    video_capture = cv2.VideoCapture("images/test.mp4")
    
    frame_count = 0
    closed_eye_frame_count = 0
    current_frames_closed_eyes = 0
    max_frames_closed_eyes = 0
    mean_frames_closed_eyes = 0
    num_blinks = 0
    previous_frame_eye_state = None
    ear_values = []

    valid_frame, frame = video_capture.read()
    while valid_frame and frame_count < max_num_frames:
        frame_count += 1
        frame_metrics = process_frame(frame)

        ear_values.append(frame_metrics["ear"])
        if frame_metrics["open_eyes"]:
            current_eye_state = "open"
            closed_eye_frame_count += 1

            if current_frames_closed_eyes > max_frames_closed_eyes:
                max_frames_closed_eyes = current_frames_closed_eyes

            current_frames_closed_eyes = 0
        else:
            current_eye_state = "closed"
            current_frames_closed_eyes += 1

        if previous_frame_eye_state == "open" and current_eye_state == "closed":
            num_blinks += 1

        blink_frequency = num_blinks / frame_count
        perclos = closed_eye_frame_count / frame_count
        valid_frame, frame = video_capture.read()
        previous_frame_eye_state = current_eye_state
elif input_type == "static_image":
    input_image = cv2.imread("images/yawn/34.jpg")
    process_frame(input_image)

#TODO: a la hora de determinar los tests, podemos usar metricas de falsos positivos, 
# falsos negativos y tener preferencia por los falsos positivos --> es mejor detectar un drowsy cuando no es cierto que al reves