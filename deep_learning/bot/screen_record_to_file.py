import numpy as np
import cv2
import os
import tensorflow as tf
from mss import mss
from PIL import Image

# model = tf.keras.models.load_model(r'D:\Dokumente\Uni\Signalverarbeitung\Project\model\rotordet_net_v2')
# model.summary()

monitor = {'top': 0, 'left': 0, 'width': 720, 'height': 1280}
target_height = 720
target_width = 1280
sct = mss()
while(True):
    sct_img = sct.grab(monitor)
    img = np.asarray(sct_img)
    # img_resized = tf.image.resize_with_pad(img, target_height, target_width)
    # img_resized = tf.cast(img_resized, dtype=tf.int32)
    # sample_input = np.array([np.asarray(img_resized[:, :, :3]),])
    # predicted = model.predict(sample_input)
    # i_class_pred = np.argmax(predicted)
    # os.system('cls')
    # print(i_class_pred)
    # cv2.imshow('test', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break