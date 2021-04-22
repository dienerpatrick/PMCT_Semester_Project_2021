import numpy as np
import os
import tensorflow as tf
import random


def import_data(batch_size):

    ap_train_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data/AP_Sorted/AP_Train')
    l_train_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data/L_Sorted/L_Train')

    shuffle_seed = random.randint(111, 999)

    ap_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ap_train_dir,
        validation_split=0.2,
        subset="training",
        seed=shuffle_seed,
        image_size=(512, 512),
        color_mode="grayscale",
        batch_size=None)

    ap_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        ap_train_dir,
        validation_split=0.2,
        subset="validation",
        seed=shuffle_seed,
        image_size=(512, 512),
        color_mode="grayscale",
        batch_size=None)

    l_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        l_train_dir,
        validation_split=0.2,
        subset="training",
        seed=shuffle_seed,
        image_size=(512, 512),
        color_mode="grayscale",
        batch_size=None)

    l_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        l_train_dir,
        validation_split=0.2,
        subset="validation",
        seed=shuffle_seed,
        image_size=(512, 512),
        color_mode="grayscale",
        batch_size=None)

    train_labels = np.concatenate([y for x, y in ap_train_ds], axis=0)  # .astype('float32').reshape((-1,2))
    val_labels = np.concatenate([y for x, y in ap_val_ds], axis=0)  # .astype('float32').reshape((-1,2))
    # print(train_labels.shape)

    ap_train_images = np.squeeze(np.concatenate([x for x, y in ap_train_ds], axis=0))
    l_train_images = np.squeeze(np.concatenate([x for x, y in l_train_ds], axis=0))
    ap_val_images = np.squeeze(np.concatenate([x for x, y in ap_val_ds], axis=0))
    l_val_images = np.squeeze(np.concatenate([x for x, y in l_val_ds], axis=0))

    train_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": ap_train_images, "input_2": l_train_images},
                                                        train_labels))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": ap_val_images, "input_2": l_val_images},
                                                        val_labels))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    # print(ap_train_images[0])

    return train_dataset, val_dataset
