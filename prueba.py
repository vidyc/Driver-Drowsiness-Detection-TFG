import onnx
import onnxruntime
from onnx import numpy_helper
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
import yaml
# from test_environment import TestEnvironment
# from PIL import Image

model = "public/open-closed-eye-0001/open-closed_eye.onnx"
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

closed = "s0001_00001_0_0_0_0_0_01.png"
file = "images/mrlEyes_2018_01/s0001/s0001_03094_0_1_1_2_0_01.png"

#file = "output_eyes/eyeNone.jpg"
image = cv2.imread(file)
print(image)
image_float = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
image_float = cv2.resize(image_float, dsize=(32, 32), interpolation=cv2.INTER_AREA)
print(image_float)
image_tensor = tf.convert_to_tensor(image_float, dtype=tf.float32)
image_tensor = tf.transpose(image_tensor, [2, 0, 1])
#image_tensor = tf.expand_dims(image_tensor, axis=0)
image_float = np.asarray(image_tensor)
print(image_float)

#print(image_float.shape)
# print(session.get_inputs()[0])
# result = session.run([output_name], {input_name: image_float})
# prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
# print(result)
# prediction_dict = {0: "open", 1: "closed"}
# print(prediction_dict[prediction])