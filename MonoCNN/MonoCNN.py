import numpy as np
import os
import PIL
import PIL.Image
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
import random
import sklearn.metrics
import seaborn as sn

def mono_cnn(batch_size=16, epochs=40, plot=False, confusion_matrix=True):

  data_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data/AP_Sorted_4/AP_Train')
  log_dir = os.path.join(os.getcwd(), '..', 'LOGS')

  SHUFFLE_SEED = random.randint(111, 999)
  num_classes = 4
  EPOCHS = epochs
  BATCHES = batch_size

  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed= SHUFFLE_SEED,
    image_size=(512, 512),
    color_mode="grayscale",
    batch_size=BATCHES)

  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=SHUFFLE_SEED,
    image_size=(512, 512),
    color_mode="grayscale",
    batch_size=BATCHES)

  data_augmentation = keras.Sequential(
    [
      layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(512, 512, 1)),
      layers.experimental.preprocessing.RandomRotation(0.1),
      layers.experimental.preprocessing.RandomZoom(0.1),
    ]
  )

  model = tf.keras.Sequential([
    # data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(512, 512, 1)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(128),
    layers.Activation('relu'),
    layers.Dense(num_classes),
    layers.Activation('softmax')
  ])


  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback]
  )


  if confusion_matrix:
    pred = model.predict(val_ds, batch_size=batch_size)
    pred_argmax = np.array([np.argmax(i) for i in pred])
    val_labels = np.concatenate([y for x, y in val_ds], axis=0).astype('float32')
    val_labels_argmax = np.array([np.argmax(i) for i in val_labels])

    cf = sklearn.metrics.confusion_matrix(val_labels_argmax, pred_argmax)

    plt.figure(figsize=(10, 7))
    sn.heatmap(cf, annot=True)
    plt.show()

  if plot:
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
    plt.savefig('plot.png')
    plt.show()


mono_cnn(epochs=20)