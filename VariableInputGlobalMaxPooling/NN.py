# source: https://medium.com/mindboard/image-classification-with-variable-input-resolution-in-keras-cbfbe576126f
#imports

import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow_addons.layers import SpatialPyramidPooling2D

import numpy as np
from cv2 import resize
from os import path, listdir
import os


# set up MobileNet GlobalMaxPooling and unsepcified input resolution

data_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data')

SHUFFLE_SEED = 69420

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed= SHUFFLE_SEED,
  image_size=(512, 512),
  color_mode="grayscale",
  batch_size=16)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=SHUFFLE_SEED,
  image_size=(512, 512),
  color_mode="grayscale",
  batch_size=16)

inputs = Input(shape=(None,None,3))
net = MobileNetV2(include_top=False, alpha=0.35, weights='imagenet', input_tensor=inputs, classes=n_classes)
net = SpatialPyramidPooling2D()(net.output)
outputs = Dense(2, activation='softmax')(net)

model = Model(inputs=inputs,outputs=outputs)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
