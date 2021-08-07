import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
import random
from matplotlib import pyplot as plt

def model_predict(model,img,target_height,target_width):

    img = np.asarray(img)
    img_resized = tf.image.resize_with_pad(img, target_height, target_width)
    img_resized = tf.cast(img_resized, dtype=tf.int32)
    sample_input = np.array([np.asarray(img_resized[:, :, :3]),])
    predicted = model.predict(sample_input)
    i_class_pred = np.argmax(predicted)

    return i_class_pred, predicted

model = tf.keras.models.load_model(r'D:\Dokumente\Uni\Signalverarbeitung\Project\model\rotordet_net_v4')
config = model.get_config() 
batch_shape = config["layers"][0]["config"]["batch_input_shape"]
target_height = batch_shape[1]
target_width = batch_shape[2]

class_names = ['DL1','DL2','DL3','DL_BG']

# input_folder = r'D:\Dokumente\Uni\Signalverarbeitung\Project\data\SignalVerarb_videos\baseline\frames\drinnen_1280x720_50fps'
input_folder = r'D:\Dokumente\Uni\Signalverarbeitung\Project\data\SignalVerarb_videos\640x360\frames\drinnen_640x360_50fps'
all_imgs = os.listdir(input_folder)
all_imgs = [os.path.join(input_folder,x) for x in all_imgs]
n_imgs = len(all_imgs)
sample_imgs = random.sample(range(n_imgs),100)
sample_imgs = [all_imgs[i] for i in sample_imgs]

for img_sample in all_imgs:
    img = tf.keras.preprocessing.image.load_img(img_sample, target_size=(target_height, target_width))
    i_class_pred, predicted = model_predict(model,img,target_height,target_width)
    class_name_pred = class_names[i_class_pred]
    # print('Class predicted: ' + class_name_pred)
    # print(predicted)
    plt.imshow(img)
    plt.title('Class predicted: ' + class_name_pred + ': ' + str(np.round(predicted[0],4)))
    plt.waitforbuttonpress()
    