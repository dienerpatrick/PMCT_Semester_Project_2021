import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfds
from Dual_InputCNN.DataImport import import_data
from tensorflow.keras.utils import plot_model


class BinaryCNN:

    def __init__(self, batch_size, epochs):

        self.epochs = epochs

        self.train_images, self.val_images, self.train_labels, self.val_labels = import_data(batch_size=batch_size)

        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        self.preprocess_input = tf.keras.applications.resnet50.preprocess_input

        self.ap_base_model = ResNet50(input_shape=(512, 512, 3),
                                      include_top=False,
                                      weights="imagenet")

        # self.l_base_model = ResNet50(input_shape=(512, 512, 3),
        #                              include_top=False,
        #                              weights="imagenet")

        ##############################################################
        #                       FREEZE MODEL                        #
        ##############################################################

        # self.l_base_model.trainable = False
        self.ap_base_model.trainable = False

        ##############################################################
        #                       DEFINE LAYERS                        #
        ##############################################################

        self.concatenate_layer = layers.Concatenate(axis=1)
        self.global_average_layer = layers.GlobalAveragePooling2D()
        self.prediction_layer = layers.Dense(2)

        ##############################################################
        #                       BUILD MODEL                          #
        ##############################################################

        self.input_x = keras.Input(shape=(512, 512, 3))
        self.input_y = keras.Input(shape=(512, 512, 3))


        self.x = self.data_augmentation(self.input_x)
        self.x = self.preprocess_input(self.x)
        self.x = self.ap_base_model(self.x, training=False)
        self.x = self.global_average_layer(self.x)

        self.y = self.data_augmentation(self.input_y)
        self.y = self.preprocess_input(self.y)
        self.y = self.ap_base_model(self.y, training=False)
        self.y = self.global_average_layer(self.y)

        self.z = self.concatenate_layer([self.x, self.y])
        self.z = layers.Dropout(0.2)(self.z)
        # self.z = layers.Dense(128, activation='relu')(self.z)
        self.outputs = self.prediction_layer(self.z)

        self.model = keras.Model(inputs=[self.input_x, self.input_y], outputs=self.outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])


        # plot_model(self.model, to_file='model.png')

    def fit(self):
        self.history = self.model.fit(x=self.train_images,
                                      y=self.train_labels,
                                      epochs=self.epochs,
                                      validation_data=(self.val_images, self.val_labels))


TestNetwork = BinaryCNN(batch_size=32, epochs=5)

TestNetwork.fit()