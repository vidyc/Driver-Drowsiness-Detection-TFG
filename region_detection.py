import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import openvino.runtime as ov
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import time
import utils
from head_pose_model import SixDRepNet
# import dlib

import math

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_cascade = cv2.CascadeClassifier("cascade_detectors/haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("cascade_detectors/haarcascade_eye_tree_eyeglasses.xml")

# dlib_face_detector = dlib.get_frontal_face_detector()
# dlib_landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
dnn_face_detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

core = ov.Core()
alternative_landmark_model = core.compile_model("landmark_model/mobilefacenet.xml", "AUTO")
alternative_landmark_model_input_size = (112, 112)
# alternative_landmark_model = tf.saved_model.load("pose_model") 


head_pose_model = SixDRepNet(backbone_name='RepVGG-B1g2',
                    backbone_file='',
                    deploy=True,
                    pretrained=False)
saved_state_dict = torch.load("6DRepNet_300W_LP_BIWI.pth")
if 'model_state_dict' in saved_state_dict:
    head_pose_model.load_state_dict(saved_state_dict['model_state_dict'])
else:
    head_pose_model.load_state_dict(saved_state_dict)

head_pose_model.eval()
head_pose_model.cuda()
head_pose_model_transformations = transforms.Compose([transforms.Resize(224),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def opencv_detect_faces(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)

        faceROI = gray[y:y+h,x:x+w]
        # eye detection
        eyes = eye_cascade.detectMultiScale(faceROI)
        num_open_eyes = 0
        for (x2, y2, w2, h2) in eyes:
            #cv2.rectangle(img, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (0, 255, 0), 2)
            new_y = y2 + int(h2/4)
            new_h = int(h2/2)
            cv2.rectangle(img_copy, (x+x2, y+y2+int(h2/4)), (x+x2+w2, y+y2+int(3*h2/4)), (0, 255, 0), 2)

            gray_eye_image = gray[y+new_y:y+new_y+new_h, x+x2:x+x2+w2]
            width, height = gray_eye_image.shape
            imS = cv2.resize(gray_eye_image, (width*4, height*4))                # Resize image
            cv2.imshow('eye', imS)

    cv2.imshow('img', img_copy)
    cv2.waitKey()


def mediapipe_face_detection(input_image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    
        annotated_image = input_image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
        return annotated_image, results


def mediapipe_face_mesh(input_image, debug=False):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1, min_tracking_confidence=0.5) as face_mesh:
        input_image.flags.writeable = False

        faces = face_mesh.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        annotated_image = input_image.copy()
        if faces is not None and faces.multi_face_landmarks is not None:
            if debug:
                for face_landmarks in faces.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())

        return faces, annotated_image


def dlib_face_landmarks(input_image, drawn_image, debug=False, landmarks=False):
    faces_dlib = dlib_face_detector(input_image, 1)
    # drawn_image = input_image.copy()

    if len(faces_dlib) > 0:
        x = faces_dlib[0].left()
        y = faces_dlib[0].top()
        x1 = faces_dlib[0].right()
        y1 = faces_dlib[0].bottom()

        drawn_image = cv2.rectangle(drawn_image, (x, y), (x1, y1), (0, 0, 255), 2)
        
        if landmarks:
            landmark_tuple = []
            for k, d in enumerate(faces_dlib):
                landmarks = dlib_landmark_detector(input_image, d)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmark_tuple.append((x, y))
                    cv2.circle(drawn_image, (x, y), 1, (255, 255, 0), -1)
        
        if debug:
            cv2.imshow("dlib", drawn_image)
            cv2.waitKey()
    else:
        print("DLIB DID NOT FIND A FACE!!!")
    
    return drawn_image


def dnn_face_detection(input_image, drawn_image, debug=False):
    h, w = input_image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(input_image, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    dnn_face_detector.setInput(blob)
    faces = dnn_face_detector.forward()
    #to draw faces on image
    # drawn_image_dnn = input_image.copy()
    faces_dnn = []
    
    found_face = False
    max_confidence = -1
    best_box = ()

    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            found_face = True
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            if confidence > max_confidence:
                max_confidence = confidence
                best_box = (x, y, x1, y1)

    if debug:
        cv2.imshow("dnn", drawn_image)
        cv2.waitKey()
    
    if found_face:
        (x, y, x1, y1) = best_box
        drawn_image = cv2.rectangle(drawn_image, (x, y), (x1, y1), (255, 0, 0), 2)
    
    return found_face, best_box, drawn_image


def estimate_alternative_landmarks(input_image, facebox, drawn_image, debug=False):
    height, width, _ = input_image.shape
    image = input_image.copy()

    x = max(facebox[0], 0)
    x1 = min(facebox[2], width)
    y = max(facebox[1], 0)
    y1 = min(facebox[3], height)
    facebox = (x, y, x1, y1)

    face_img = image[facebox[1]: facebox[3],
                     facebox[0]: facebox[2]]
    face_height, face_width, _ = face_img.shape                     
    infer_request = alternative_landmark_model.create_infer_request()

    inp_image = cv2.resize(face_img.copy(), dsize=(112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
    inp_image = inp_image / 255.0
    inp_image = inp_image.transpose((2, 0, 1))
    inp_image = tf.expand_dims(inp_image, axis=0)
    inp_image = np.asarray(inp_image)
    # inp_image = inp_image.reshape((1,) + inp_image.shape) 
    
    input_tensor = ov.Tensor(array=inp_image, shared_memory=True)
    infer_request.set_input_tensor(input_tensor)

    infer_request.start_async()
    infer_request.wait()

    output = infer_request.get_output_tensor()
    landmarks = output.data.reshape(-1, 2)

    for landmark in landmarks:
        x = landmark[0] * face_width + facebox[0]
        y = landmark[1] * face_height + facebox[1]
        cv2.circle(drawn_image, (int(x), int(y)), 1, (0, 255, 0), 0, cv2.LINE_AA)

    normalized_landmarks = [ [(landmark[0] * face_width + facebox[0])/width, (landmark[1] * face_height + facebox[1])/height] for landmark in landmarks ]

    if debug:
        cv2.imshow('alt_landmarks', drawn_image)
        cv2.waitKey()

    return landmarks, drawn_image


def draw_head_pose_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    

    # # Draw all the lines
    # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    k = (point_2d[5] + point_2d[8])//2
    # cv2.line(img, tuple(point_2d[1]), tuple(
    #     point_2d[6]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[2]), tuple(
    #     point_2d[7]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[3]), tuple(
    #     point_2d[8]), color, line_width, cv2.LINE_AA)
    
    return(point_2d[2], k)


def estimate_head_pose_model(face_img, facebox, debug=False):
    img = Image.fromarray(face_img)
    img = img.convert('RGB')
    img = head_pose_model_transformations(img)

    img = torch.Tensor(img[None, :]).cuda(0)

    start = time.time()
    R_pred = head_pose_model(img)
    end = time.time()

    euler = utils.compute_euler_angles_from_rotation_matrices(
        R_pred)*180/np.pi
    pitch = euler[:, 0].cpu().detach().numpy()[0] + 90
    yaw = euler[:, 1].cpu().detach().numpy()[0]
    roll = euler[:, 2].cpu().detach().numpy()[0]

    return {"yaw": yaw, "pitch": pitch, "roll": roll}, face_img


def estimate_head_pose(input_image, drawn_image, landmarks, debug=False):
    size = input_image.shape
    # drawn_image = input_image.copy()

    points_2D_indexes = [1, 199, 263, 33, 291, 61]
    points_2D = np.array([(landmarks[index].x * size[1], landmarks[index].y * size[0]) for index in points_2D_indexes])
    points_3D = np.array([
            (0.0, 0.0, 0.0),          #Nose tip
            (0.0, -330.0, -65.0),     #Chin
            (-225.0, 170.0, -135.0),  #Left eye corner
            (225.0, 170.0, -135.0),   #Right eye corner 
            (-150.0, -150.0, -125.0), #Left mouth 
            (150.0, -150.0, -125.0)   #Right mouth 
            ])  
    
    dist_coeffs = np.zeros((4,1))  
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    
    success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    p1 = ( int(points_2D[0][0]), int(points_2D[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    x1, x2 = draw_head_pose_box(drawn_image, rotation_vector, translation_vector, camera_matrix)

    try:
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]
        z = 1
        pitch = int(math.degrees(math.acos(y / math.sqrt(x**2 + y**2 + z**2))))
        # pitch = int(math.degrees(math.atan(y/x))) + 90
    except:
        pitch = 90
    
    try:
        x = x2[0] - x1[0]
        y = x2[1] - x1[1]
        z = 1
        yaw = int(math.degrees(math.acos(x / math.sqrt(x**2 + y**2 + z**2)))) - 90 # shiftamos 90 grados para que recto = 0
        # yaw = int(math.degrees(math.atan(-x/y)))
    except:
        yaw = 90

    for point in points_2D:
        x = int(point[0])
        y = int(point[1])
        drawn_image = cv2.circle(drawn_image, (x, y), 2, (255, 255, 0), -1)

    drawn_image = cv2.line(drawn_image, p1, p2, (255,255,255), 2)
    if debug:
        cv2.imshow('head_pose', drawn_image)
        cv2.waitKey()
    
    angle_dict = {"yaw": yaw, "pitch": pitch}

    return angle_dict, drawn_image


def get_cropped_image_from_landmarks(img, face_dimensions, landmark_coordinates, landmarks):

    landmark_list = [ [ landmark_coordinates[landmark].x for landmark_pair in landmarks for landmark in landmark_pair ],
                      [ landmark_coordinates[landmark].y for landmark_pair in landmarks for landmark in landmark_pair ] ]
    

    height, width, _ = img.shape
    face_width, face_height = face_dimensions
    eye_height = face_height // 7   # 14% de la altura de la cara
    eye_width  = int(face_width * 0.3)     # 30% de la anchura de la cara

    top = int(min(landmark_list[1]) * height)
    left = int(min(landmark_list[0]) * width)
    right = int(max(landmark_list[0]) * width)
    bottom = int(max(landmark_list[1]) * height)

    min_distance = 10
    vertical_distance = bottom - top
    horizontal_distance = right - left

    remaining_eye_height = (eye_height - vertical_distance)//2
    remaining_eye_width = (eye_width - horizontal_distance)//2

    top = max(top - remaining_eye_height, 0)
    bottom = min(bottom + remaining_eye_height, height)

    left = max(left - remaining_eye_width, 0)
    right = min(right + remaining_eye_width, width)

    vertical_distance = bottom - top
    horizontal_distance = right - left

    # if vertical_distance < min_distance:
    #     top -= min_distance // 2
    #     bottom += min_distance // 2
    
    # if horizontal_distance < min_distance:
    #     left -= min_distance // 2
    #     right += min_distance // 2

    cropped_eye = img[int(top):int(bottom), int(left):int(right)]
    resized_cropped_eye = cv2.resize(cropped_eye, (52, 52), interpolation=cv2.INTER_CUBIC)
    return resized_cropped_eye


def get_ROI_images(image, face_dimensions, landmark_coordinates):

    ROI_images_dict = {}

    left_eye = get_cropped_image_from_landmarks(image, face_dimensions, landmark_coordinates, mp_face_mesh.FACEMESH_LEFT_EYE)
    ROI_images_dict["left_eye"] = left_eye

    right_eye = get_cropped_image_from_landmarks(image, face_dimensions, landmark_coordinates, mp_face_mesh.FACEMESH_RIGHT_EYE)
    ROI_images_dict["right_eye"] = right_eye

    return ROI_images_dict

def get_iris_metrics(image, face_landmarks, iris_indexes):

    iris_metrics = {}
    iris_metrics["centers"] = get_iris_centers(image, face_landmarks, iris_indexes)
    iris_metrics["diameters"] = get_iris_diameters(image, face_landmarks, iris_indexes)
    return iris_metrics

def get_iris_centers(image, face_landmarks, iris_indexes):

    height, width, _ = image.shape
    iris_centers = {}

    for iris, indexes in iris_indexes.items():
        normalized_iris_center = face_landmarks[indexes[0]]
        iris_center = (normalized_iris_center.x * width, normalized_iris_center.y * height)
        iris_centers[iris] = iris_center
    
    return iris_centers

def get_iris_diameters(image, face_landmarks, iris_indexes):
    height, width, _ = image.shape
    iris_diameters = {}

    ## left
    top_left = face_landmarks[470].y
    bot_left = face_landmarks[472].y
    iris_diameters["left"] = (bot_left - top_left) * height

    ## right
    top_right = face_landmarks[475].y
    bot_right = face_landmarks[477].y
    iris_diameters["right"] = (bot_right - top_right) * height
    
    return iris_diameters