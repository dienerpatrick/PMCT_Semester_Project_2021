import numpy as np
import os
import PIL
import PIL.Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random


data_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data/AP_Sorted_Bin_NoTest')

SHUFFLE_SEED = random.randint(111, 999)
num_classes = 2
EPOCHS = 10

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed= SHUFFLE_SEED,
  image_size=(512, 512),
  color_mode="grayscale",
  batch_size=32)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=SHUFFLE_SEED,
  image_size=(512, 512),
  color_mode="grayscale",
  batch_size=32)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(512, 512, 1)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = tf.keras.Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(512, 512, 1)),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  # layers.Dropout(0.3),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  # layers.Dropout(0.3),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  # layers.Dropout(0.3),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Flatten(),
  # layers.Dense(256, activation='relu'),
  layers.Dense(128),
  layers.Activation('relu'),
  layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

print(f'SHUFFLE SEED: {SHUFFLE_SEED}')

model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title(f'Training and Validation Accuracy with Seed: {SHUFFLE_SEED}')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()