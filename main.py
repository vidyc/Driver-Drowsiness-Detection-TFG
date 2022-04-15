import random

import pandas as pd
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from test_environment import TestEnvironment
from models import naive_model

max_num_frames = 5400

if __name__ == "__main__":
    test_environment = TestEnvironment()

    if True:
        path = "images/Fold3_part2/31/"
        #df = naive_model.create_dataset_from_videos(path)
        df = pd.read_csv("test_dataset.csv")
        data = df.drop("label", axis = 1)
        labels = df["label"]
        x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.4, random_state=1 )
        knn = KNeighborsClassifier(n_neighbors=30)
        knn.fit(x_train, y_train)
        knn.predict(x_test)
        #print(naive_model.knn(df, [[0.23, 0.05, 50, 0.04, 0], [0.23, 0.05, 50, 0.04, 0]], 10))

    if False:
        image = cv2.imread("images/test2.jpg")
        naive_model.process_frame(image)

    if False:
        results = test_environment.test_open_close_eye_detection(naive_model)
        print(results["predictions"])
        print(results["performance_metrics"])

    if False:
        input_video = cv2.VideoCapture("images/Fold3_part2/31/10.mp4")
        
        dummy_labels = [ 10 for _ in range(0, 17004)]
        videos, labels = test_environment.prepare_NTHUDDD_dataset()
        
        results = test_environment.run_test(naive_model, input_video, dummy_labels)

        print("ANSWERS")
        print(results["predictions"])
        print(dummy_labels)
        print(results["performance_metrics"])

#TODO: a la hora de determinar los tests, podemos usar metricas de falsos positivos, 
# falsos negativos y tener preferencia por los falsos positivos --> es mejor detectar un drowsy cuando no es cierto que al reves
