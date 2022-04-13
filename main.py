import cv2
import random

from test_environment import TestEnvironment
from models import naive_model

max_num_frames = 5400

if __name__ == "__main__":
    test_environment = TestEnvironment()

    if False:
        results = test_environment.test_open_close_eye_detection(naive_model)
        print(results["predictions"])
        print(results["performance_metrics"])

    if True:
        input_video = cv2.VideoCapture("images/Fold3_part2/33/0.mp4")
        
        dummy_labels = [ 10 for _ in range(0, max_num_frames)]
        videos, labels = test_environment.prepare_NTHUDDD_dataset()
        
        results = test_environment.run_test(naive_model, input_video, dummy_labels)

        print("ANSWERS")
        print(results["predictions"])
        print(dummy_labels)
        print(results["performance_metrics"])

#TODO: a la hora de determinar los tests, podemos usar metricas de falsos positivos, 
# falsos negativos y tener preferencia por los falsos positivos --> es mejor detectar un drowsy cuando no es cierto que al reves