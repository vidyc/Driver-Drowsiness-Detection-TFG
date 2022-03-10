import cv2

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier("models/haarcascade_eye_tree_eyeglasses.xml")

video = False

def detect_faces(face_detector):
    img = cv2.imread("images/149.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        faceROI = gray[y:y+h,x:x+w]
        # eye detection
        eyes = eye_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            cv2.rectangle(img, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (0, 255, 0), 2)


    cv2.imshow('img', img)
    cv2.waitKey()

if video:
    print("Future Implementation")
else:
    detect_faces(face_cascade)