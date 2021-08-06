from tensorflow.keras.applications import *
from tensorflow.keras import models
from tensorflow.keras import layers
import pathlib
import os
import shutil
import pandas as pd
from random import randrange
from tensorflow.keras import optimizers
from sklearn import model_selection
import tensorflow as tf
import numpy as np


train_dir = r'.\deep_learning\data\train\merged'
train_dir = pathlib.Path(train_dir)
test_dir = r'.\deep_learning\data\test\merged'
test_dir = pathlib.Path(test_dir)

batch_size = 32
img_height = 360
img_width = 640
input_shape=(img_height, img_width, 3)

seed = randrange(1000)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split = 0.6, # only use 60% of the training data (for memory reasons)
  label_mode = 'categorical',
  subset = 'validation',
  seed = seed,
  smart_resize = True,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  validation_split = 0.2, # use 20% of the test set for validation
  label_mode = 'categorical',
  subset = 'validation',
  seed = seed,
  smart_resize = True,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
n_classes = len(class_names)
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

conv_base = EfficientNetB1(weights='imagenet',include_top=False,input_shape=input_shape)

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name='gap'))
model.add(layers.Dropout(rate=0.2,name='dropout'))
model.add(layers.Dense(n_classes,activation='softmax',name='fc_out'))
conv_base.trainable = False
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

epochs=10

history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)

model.save(r'.\deep_learning\models\rotordet_net_v4')
np.save(r'.\deep_learning\models\my_history.npy',history.history)