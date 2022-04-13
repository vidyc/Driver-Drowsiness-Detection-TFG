import mediapipe as mp
import cv2
from numpy import interp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("models/haarcascade_eye_tree_eyeglasses.xml")

def opencv_detect_faces(face_detector):
    img = cv2.imread("images/1.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        faceROI = gray[y:y+h,x:x+w]
        # eye detection
        eyes = eye_cascade.detectMultiScale(faceROI)
        num_open_eyes = 0
        for (x2, y2, w2, h2) in eyes:
            #cv2.rectangle(img, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (0, 255, 0), 2)
            new_y = y2 + int(h2/4)
            new_h = int(h2/2)
            cv2.rectangle(img, (x+x2, y+y2+int(h2/4)), (x+x2+w2, y+y2+int(3*h2/4)), (0, 255, 0), 2)

            gray_eye_image = gray[y+new_y:y+new_y+new_h, x+x2:x+x2+w2]
            width, height = gray_eye_image.shape
            imS = cv2.resize(gray_eye_image, (width*4, height*4))                # Resize image
            cv2.imshow('eye', imS)

            binarized_eye_image = cv2.adaptiveThreshold(gray_eye_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            width, height = binarized_eye_image.shape
            imS = cv2.resize(binarized_eye_image, (width*4, height*4))                # Resize image
            cv2.imshow('bin_eye', imS)
            cv2.waitKey(0)                                  # Display the image infinitely until any keypress

            cv2.calcHist([gray_eye_image], [0], None, [256], [0,256])
            
            mean = np.mean(gray_eye_image)
            print(f"eye_mean: {mean}")

            gray_eye_image_float = gray_eye_image.astype(np.float32)
            gray_eye_image_float_squared = gray_eye_image_float * gray_eye_image_float
            mean_float_squared = np.mean(gray_eye_image_float_squared)
            variance = mean_float_squared - mean**2
            stdev = math.sqrt(variance)
            print(f"variance: {variance}")
            print(f"standard dev: {stdev}")

            if variance > ( mean * 9 ):
                num_open_eyes += 1

            #plt.hist(gray_eye_image.ravel(), 256, [0,256])
            #plt.show()

        # PRIMERA IDEA A IMPLEMENTAR:
        # EYE BLINK METHOD --> CONTAR NUMERO PARPADEOS POR MINUTO
        # %DROWSINESS = (NUM_FRAMES_WITH_CLOSED_EYES / NUM_FRAMES) * 100
        
        num_eyes = len(eyes)
        if num_eyes == 2 and num_open_eyes == 0:
            print(f"Closed eyes!")
        else:
            print(f"Open Eyes!")

    cv2.imshow('img', img)
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


def mediapipe_face_mesh(input_image):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

        annotated_image = input_image.copy()
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
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
        return annotated_image, results


def get_cropped_image_from_landmarks(img, landmark_coordinates, landmarks):

    landmark_list = [ [ landmark_coordinates.landmark[landmark].x for landmark_pair in landmarks for landmark in landmark_pair ],
                      [ landmark_coordinates.landmark[landmark].y for landmark_pair in landmarks for landmark in landmark_pair ] ]
    

    height, width, _ = img.shape

    top = int(min(landmark_list[1]) * height)
    left = int(min(landmark_list[0]) * width)
    right = int(max(landmark_list[0]) * width)
    bottom = int(max(landmark_list[1]) * height)

    min_distance = 10
    vertical_distance = bottom - top
    horizontal_distance = right - left

    if vertical_distance < min_distance:
        top -= min_distance // 2
        bottom += min_distance // 2
    
    if horizontal_distance < min_distance:
        left -= min_distance // 2
        right += min_distance // 2

    cropped_eye = img[top:bottom, left:right]
    resized_cropped_eye = cv2.resize(cropped_eye, (227, 227), interpolation=cv2.INTER_CUBIC)
    return resized_cropped_eye


def get_ROI_images(image, landmark_coordinates):

    ROI_images_dict = {}

    left_eye = get_cropped_image_from_landmarks(image, landmark_coordinates, mp_face_mesh.FACEMESH_LEFT_EYE)
    ROI_images_dict["left_eye"] = left_eye

    right_eye = get_cropped_image_from_landmarks(image, landmark_coordinates, mp_face_mesh.FACEMESH_RIGHT_EYE)
    ROI_images_dict["right_eye"] = right_eye

    return ROI_images_dict


def compare(item):
    return item[2]

def plot_landmark(img, landmarks, facial_area_obj):    
    res_img = img.copy()
    
    height, width, _ = img.shape
    landmark_list_x = [ int(landmarks.landmark[id].x * width) for id, _ in list(facial_area_obj) ]
    landmark_list_y = [ int(landmarks.landmark[id].y * height) for id, _ in list(facial_area_obj) ]

    left = min(landmark_list_x)
    right = max(landmark_list_x)
    left_ind = landmark_list_x.index(left)
    right_ind = landmark_list_x.index(right)
    eye_width = right - left
    middle_point = left + eye_width/2

    # get 6 horizontally nearest points to middle_point
    distance_array = [ (idx, x_point, abs(middle_point - x_point)) for idx, x_point in enumerate(landmark_list_x) ]
    distance_array.sort(key=compare)
    nearest_points = [ landmark_list_y[idx] for idx, _, _ in distance_array[:6] ]

    top = min(nearest_points)
    bottom = max(nearest_points)
    top_ind = landmark_list_y.index(top)
    bottom_ind = landmark_list_y.index(bottom)
    eye_height = bottom - top

    print((top, left, right, bottom))
    indices = (top_ind, left_ind, right_ind, bottom_ind)
    print(indices)

    for ind in indices:
        x = int(landmark_list_x[ind])
        y = int(landmark_list_y[ind])
        point = (x, y)

        #res_img = cv2.circle(res_img, point, radius=4, color=(0, 0, 255), thickness=-1)
        #res_img = cv2.putText(res_img, f"{ind}", point, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4)

    print(eye_width)
    print(eye_height)

    perclos = eye_height / eye_width
    print(perclos)

    print()
    ind = 0

    print(list(facial_area_obj))

    for source_idx, target_idx in list(facial_area_obj):
        source = landmarks.landmark[source_idx]
        target = landmarks.landmark[target_idx]
 
        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

        point = ((relative_source[0]+relative_target[0]) // 2, (relative_source[1]+relative_target[1]) // 2)
        point = relative_target
        #res_img = cv2.line(res_img, relative_source, relative_target, (255, 255, 255), thickness = 2)
        #res_img = cv2.circle(res_img, relative_source, radius=4, color=(0, 0, 255), thickness=-1)
        #res_img = cv2.putText(res_img, f"{source_idx}", relative_source, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)
        #res_img = cv2.circle(res_img, relative_target, radius=4, color=(0, 0, 255), thickness=-1)
        #res_img = cv2.putText(res_img, f"{target_idx}", relative_target, color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3)
        ind+=1
    return res_img