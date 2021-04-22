import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfds



class BinaryCNN:
    def __init__(self):

        self.sorted_data_dir = "C:/Users/diene/myCloud/UZH/Biologie & Computational Science FS21/" \
                        "BIO394 Interdisciplinary Research Methods in Computational Biology/" \
                        "Post Mortem CT Image Analysis/DATA/Sorted_Data"

        self.data_dir = "C:/Users/diene/myCloud/UZH/Biologie & Computational Science FS21/" \
                               "BIO394 Interdisciplinary Research Methods in Computational Biology/" \
                               "Post Mortem CT Image Analysis/DATA"


        self.ap_train = tf.keras.preprocessing.image_dataset_from_directory(
            self.sorted_data_dir + "/AP_sorted",
            validation_split=0.2,
            subset="training",
            seed=321,
            image_size=(512, 512),
            color_mode='rgb',
            batch_size=32)

        print(type(self.ap_train))

        self.ap_val = tfds.as_numpy(tf.keras.preprocessing.image_dataset_from_directory(
            self.sorted_data_dir + "/AP_sorted",
            validation_split=0.2,
            subset="validation",
            seed=321,
            image_size=(512, 512),
            color_mode='rgb',
            batch_size=32))

        self.l_train = tfds.as_numpy(tf.keras.preprocessing.image_dataset_from_directory(
            self.sorted_data_dir + "/L_sorted",
            validation_split=0.2,
            subset="training",
            seed=321,
            image_size=(512, 512),
            color_mode='rgb',
            batch_size=32))

        self.l_val = tfds.as_numpy(tf.keras.preprocessing.image_dataset_from_directory(
            self.sorted_data_dir + "/L_sorted",
            validation_split=0.2,
            subset="validation",
            seed=321,
            image_size=(512, 512),
            color_mode='rgb',
            batch_size=32))

        self.data_augmentation = tf.keras.Sequential([
                                    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                    ])

        self.preprocess_input = tf.keras.applications.resnet50.preprocess_input

        self.ap_base_model = ResNet50(input_shape=(512, 512, 3),
                                      include_top=False,
                                      weights="imagenet")

        self.l_base_model = ResNet50(input_shape=(512, 512, 3),
                                     include_top=False,
                                     weights="imagenet")

        ##############################################################
        #                       FREEZE MODEL                        #
        ##############################################################

        self.l_base_model.trainable = False
        self.ap_base_model.trainable = False

        ##############################################################
        #                       DEFINE LAYERS                        #
        ##############################################################

        self.concatenate_layer = layers.Concatenate(axis=0)
        self.global_average_layer = layers.GlobalAveragePooling2D()
        self.prediction_layer = layers.Dense(1)

        ##############################################################
        #                       BUILD MODEL                          #
        ##############################################################

        self.input_x = keras.Input(shape=(512, 512, 3))
        # self.input_y = keras.Input(shape=(512, 512, 3))

        self.x = self.data_augmentation(self.input_x)
        self.x = self.preprocess_input(self.x)
        self.x = self.ap_base_model(self.x, training=False)
        self.x = self.global_average_layer(self.x)

        # self.y = self.data_augmentation(self.input_y)
        # self.y = self.preprocess_input(self.y)
        # self.y = self.ap_base_model(self.y, training=False)
        # self.y = self.global_average_layer(self.y)
        #
        # self.z = self.concatenate_layer([self.x, self.y])
        # self.z = layers.Dropout(0.2)(self.z)

        self.outputs = self.prediction_layer(self.x)
        self.model = keras.Model(self.input_x, self.outputs)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])


    def fit(self):

        self.history = self.model.fit(self.ap_train,
                                      epochs=10,
                                      validation_data=self.ap_val)



testCNN = BinaryCNN()

testCNN.fit()