import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def model_predict(model,img,target_height,target_width):

    img = np.asarray(img)
    img_resized = tf.image.resize_with_pad(img, target_height, target_width)
    img_resized = tf.cast(img_resized, dtype=tf.int32)
    sample_input = np.array([np.asarray(img_resized[:, :, :3]),])
    predicted = model.predict(sample_input)
    i_class_pred = np.argmax(predicted)

    return i_class_pred, predicted

RTSP_url = 'rtsp://operator:operator@192.168.0.90:554/axis-media/media.amp'

cap = cv2.VideoCapture(RTSP_url)
model = tf.keras.models.load_model(r'.\deep_learning\model\rotordet_net_v4')
config = model.get_config() 
batch_shape = config["layers"][0]["config"]["batch_input_shape"]
target_height = batch_shape[1]
target_width = batch_shape[2]

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,target_height-10)
fontScale = 1
fontColor = (0,0,255)
lineType = 2

class_names = ['DL1','DL2','DL3','DL_BG']

while(True):
    ret, img = cap.read()
    i_class_pred, predicted = model_predict(model,img,target_height,target_width)
    class_name_pred = class_names[i_class_pred]
    os.system('cls')
    print(class_name_pred)

    cv2.putText(img,class_name_pred, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.imshow('Livestream (to close press "Q")', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
