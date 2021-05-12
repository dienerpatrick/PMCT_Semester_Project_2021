import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from DATA.DataImport import import_datasets_resnet
import matplotlib.pyplot as plt


class BinaryCNN:

    def __init__(self, batch_size, epochs):

        self.epochs = epochs

        self.weight_zero = (1 / 401) * 624 / 2.0
        self.weight_one = (1 / 105) * 624 / 2.0
        self.weight_two = (1 / 43) * 624 / 2.0
        self.weight_three = (1 / 75) * 624 / 2.0

        self.weights = {0: self.weight_zero, 1: self.weight_one,
                        2: self.weight_two, 3: self.weight_three}

        self.train_ds, self.val_ds = import_datasets_resnet(batch_size=batch_size)

        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        self.preprocess_input = tf.keras.applications.resnet50.preprocess_input

        self.ap_base_model = ResNet50(input_shape=(224, 224, 3),
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
        self.prediction_layer = layers.Dense(4)

        ##############################################################
        #                       BUILD MODEL                          #
        ##############################################################

        self.input_x = keras.Input(shape=(224, 224, 3), name="input_1")
        self.input_y = keras.Input(shape=(224, 224, 3), name="input_2")


        self.x = self.data_augmentation(self.input_x)
        self.x = self.preprocess_input(self.x)
        self.x = self.ap_base_model(self.x, training=False)
        self.x = self.global_average_layer(self.x)
        self.x = keras.Model(inputs=self.input_x, outputs=self.x)

        self.y = self.data_augmentation(self.input_y)
        self.y = self.preprocess_input(self.y)
        self.y = self.ap_base_model(self.y, training=False)
        self.y = self.global_average_layer(self.y)
        self.y = keras.Model(inputs=self.input_y, outputs=self.y)

        combined = layers.concatenate([self.x.output, self.y.output])

        self.z = layers.Dropout(0.4)(combined)
        # self.z = layers.Dense(128)(self.z)
        # self.z = layers.Activation('relu')(self.z)
        self.z = layers.Dense(4)(self.z)

        self.model = keras.Model(inputs=[self.x.input, self.y.input], outputs=self.z)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])


        # plot_model(self.model, to_file='model.png')

    def fit(self):
        self.history = self.model.fit(self.train_ds,
                                      epochs=self.epochs,
                                      # callbacks=[lr_decay_callback],
                                      validation_data=self.val_ds,
                                      #class_weight=self.weights
                                      )

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

TestNetwork = BinaryCNN(batch_size=32, epochs=20)

TestNetwork.fit()