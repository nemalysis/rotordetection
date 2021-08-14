# This file has to be run from within the deep_learning folder, such that the labeled data can be accessed via .\data\labeled\ 

from tensorflow.keras.applications import *
from tensorflow.keras import models
from tensorflow.keras import layers
import pathlib
from random import randrange
import tensorflow as tf
import numpy as np

# Specify where the training and test data is stored
train_dir = r'.\data\train\merged'
train_dir = pathlib.Path(train_dir)
test_dir = r'.\data\test\merged'
test_dir = pathlib.Path(test_dir)

batch_size = 32
img_height = 360
img_width = 640
input_shape=(img_height, img_width, 3)

seed = randrange(1000)

# train_ds is an object with which the training data can be loaded.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split = 0.6, # only use 60% of the training data (for memory reasons)
  subset = 'validation', # set subset to 'validation' to use percentage specified in validation_split. Else 1 - validation_split % of the data is used.
  label_mode = 'categorical',
  seed = seed,
  smart_resize = True,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# test_ds is an object with which the test data can be loaded.
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  validation_split = 0.2, # use 20% of the test set for validation
  subset = 'validation',
  label_mode = 'categorical',
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

# Load the EfficientNetB1 with pretrained weights
conv_base = EfficientNetB1(weights='imagenet',include_top=False,input_shape=input_shape)

# Construct the model:
model = models.Sequential()
model.add(conv_base)                                                      # Add the EfficientNetB1 as convolutional base
model.add(layers.GlobalMaxPooling2D(name='gap'))                          # Add a global max pool layer
model.add(layers.Dropout(rate=0.2,name='dropout'))                        # Add a dropout layer
model.add(layers.Dense(n_classes,activation='softmax',name='fc_out'))     # Add the readout layer, i.e. 4 neuron fully connected

conv_base.trainable = False # Freeze the weights of the convolutional base

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.summary()

epochs=10

# Train the network
history = model.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)

# Save the trained network and the training history
model.save(r'.\models\rotordet_net_vX')
np.save(r'.\models\rotordet_net_vX\my_history.npy',history.history)