import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from DATA.DataImport import import_datasets_resnet
import matplotlib.pyplot as plt
import os

class MonoCNNResNet:

    def __init__(self, batch_size, epochs):
        self.log_dir = os.path.join(os.getcwd(), '..', 'LOGS')
        self.epochs = epochs

        self.weight_zero = (1 / 401) * 624 / 2.0
        self.weight_one = (1 / 105) * 624 / 2.0
        self.weight_two = (1 / 43) * 624 / 2.0
        self.weight_three = (1 / 75) * 624 / 2.0

        self.weights = {0: self.weight_zero, 1: self.weight_one,
                        2: self.weight_two, 3: self.weight_three}

        self.ap_train_ds, self.ap_val_ds = import_datasets_resnet(batch_size=batch_size, imgset='ap')

        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        self.preprocess_input = tf.keras.applications.resnet50.preprocess_input

        self.base_model = ResNet50(input_shape=(224, 224, 3),
                                   include_top=False,
                                   weights="imagenet")

        ##############################################################
        #                       FREEZE MODEL                        #
        ##############################################################

        self.base_model.trainable = False

        ##############################################################
        #                       DEFINE LAYERS                        #
        ##############################################################

        self.global_average_layer = layers.GlobalAveragePooling2D()
        self.prediction_layer = layers.Dense(4, dtype=tf.float32)

        ##############################################################
        #                       BUILD MODEL                          #
        ##############################################################

        #self.input_x = keras.Input(shape=(224, 224, 3), name="input_1")

        # self.x = self.data_augmentation(self.input_x)
        # self.x = self.preprocess_input(self.input_x)
        # self.x = self.base_model(self.x, training=False)
        # self.x = self.global_average_layer(self.x)
        # self.x = keras.layers.Flatten()(self.x)
        # self.x = keras.layers.Dropout(0.3)(self.x)
        # self.x = self.prediction_layer(self.x)

        self.x = layers.Flatten()(self.base_model.output)
        self.x = layers.Dense(1000, activation='relu')(self.x)
        self.x = layers.Dropout(0.3)(self.x)
        self.predictions = layers.Dense(4, activation='softmax')(self.x)

        self.model = keras.Model(inputs=self.base_model.input, outputs=self.predictions)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           # loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits = True),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def fit(self, plot=False):

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir)

        self.history = self.model.fit(self.ap_train_ds,
                                      epochs=self.epochs,
                                      # callbacks=[lr_decay_callback],
                                      validation_data=self.ap_val_ds,
                                      # class_weight=self.weights,
                                      callbacks=[tensorboard_callback]
                                      )

        if plot:
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']

            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            epochs_range = range(self.epochs)

            plt.figure(figsize=(8, 8))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.savefig('plot.png')
            plt.show()


TestNetwork = MonoCNNResNet(batch_size=16, epochs=40)

TestNetwork.fit()
