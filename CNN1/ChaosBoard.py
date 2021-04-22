import numpy as np
import os
import PIL
import PIL.Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

data_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data/')

SHUFFLE_SEED = random.randint(111, 999)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=SHUFFLE_SEED,
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