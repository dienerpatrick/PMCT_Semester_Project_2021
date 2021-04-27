import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Dual_InputCNN.DataImport import import_data
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
import math


class BinaryCNN:

    def __init__(self, batch_size, epochs):

        self.epochs = epochs

        self.train_ds, self.val_ds = import_data(batch_size=batch_size)

        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ])


        ##############################################################
        #                       DEFINE LAYERS                        #
        ##############################################################

        self.concatenate_layer = layers.Concatenate(axis=1)
        self.data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),
            ]
        )

        ##############################################################
        #                       BUILD MODEL                          #
        ##############################################################

        self.input_x = keras.Input(shape=(512, 512, 1), name="input_1")
        self.input_y = keras.Input(shape=(512, 512, 1), name="input_2")

        self.x = self.data_augmentation(self.input_x)
        self.x = layers.experimental.preprocessing.Rescaling(1. / 255)(self.x)
        self.x = layers.Conv2D(32, 3, activation='relu')(self.x)
        self.x = layers.MaxPooling2D()(self.x)
        self.x = layers.Dropout(0.3)(self.x)
        self.x = layers.Conv2D(32, 3, activation='relu')(self.x)
        self.x = layers.MaxPooling2D()(self.x)
        self.x = layers.Dropout(0.3)(self.x)
        self.x = layers.Conv2D(32, 3, activation='relu')(self.x)
        self.x = layers.MaxPooling2D()(self.x)
        self.x = layers.Dropout(0.3)(self.x)
        self.x = layers.Conv2D(32, 3, activation='relu')(self.x)
        self.x = layers.MaxPooling2D()(self.x)
        self.x = layers.Dropout(0.3)(self.x)
        self.x = layers.Flatten()(self.x)
        self.x = keras.Model(inputs=self.input_x, outputs=self.x)

        self.y = self.data_augmentation(self.input_y)
        self.y = layers.experimental.preprocessing.Rescaling(1. / 255)(self.y)
        self.y = layers.Conv2D(32, 3, activation='relu')(self.y)
        self.y = layers.MaxPooling2D()(self.y)
        # self.y = layers.Dropout(0.3)(self.y)
        self.y = layers.Conv2D(32, 3, activation='relu')(self.y)
        self.y = layers.MaxPooling2D()(self.y)
        # self.y = layers.Dropout(0.3)(self.y)
        self.y = layers.Conv2D(32, 3, activation='relu')(self.y)
        self.y = layers.MaxPooling2D()(self.y)
        # self.y = layers.Dropout(0.3)(self.y)
        self.y = layers.Conv2D(32, 3, activation='relu')(self.y)
        self.y = layers.MaxPooling2D()(self.y)
        self.y = layers.Dropout(0.3)(self.y)
        self.y = layers.Flatten()(self.y)
        self.y = keras.Model(inputs=self.input_y, outputs=self.y)

        combined = layers.concatenate([self.x.output, self.y.output])

        self.z = layers.Dropout(0.2)(combined)
        # self.z = layers.Dense(128, activation='relu')(self.z)
        self.z = layers.Dense(2)(self.z)

        self.model = keras.Model(inputs=[self.x.input, self.y.input], outputs=self.z)

        opt = SGD(lr=0.0001)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        def lr_decay(epoch):
            return 0.001 * math.pow(0.6, epoch)

        # lr schedule callback
        lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)

        self.history = self.model.fit(self.train_ds,
                                      epochs=self.epochs,
                                      callbacks=[lr_decay_callback],
                                      validation_data=self.val_ds)

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
        plt.title(f'Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()




TestNetwork = BinaryCNN(batch_size=16, epochs=5)