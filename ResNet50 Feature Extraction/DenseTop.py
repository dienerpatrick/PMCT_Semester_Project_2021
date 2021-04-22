import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import random


class DenseModel:
    def __init__(self, ap_data, l_data, labels):

        self.BATCH_SIZE = 50
        self.EPOCHS = 10

        self.labels = labels
        self.data = np.concatenate((ap_data, l_data), axis=1)


        self.seed = random.randint(111, 999)

        # split data into train, val and test

        self.feat_train_val, self.feat_test, \
            self.lab_train_val, self.lab_test = train_test_split(self.data, self.labels,
                                                                 test_size=0.1, random_state=self.seed)

        self.feat_train, self.feat_val, \
            self.lab_train, self.lab_val = train_test_split(self.feat_train_val, self.lab_train_val,
                                                            test_size=(2 / 9), random_state=self.seed)

        print(self.feat_train.shape)

        # generate data sets

        print(self.lab_train.shape)

        self.train_data = tf.data.Dataset.from_tensor_slices((self.feat_train, self.lab_train))
        self.val_data = tf.data.Dataset.from_tensor_slices((self.feat_val, self.lab_val))
        self.test_data = tf.data.Dataset.from_tensor_slices((self.feat_test, self.lab_test))


        self.SHUFFLE_BUFFER_SIZE = 100

        self.train_data = self.train_data.shuffle(self.SHUFFLE_BUFFER_SIZE).batch(self.BATCH_SIZE)
        self.val_data = self.val_data.batch(self.BATCH_SIZE)

        self.model = tf.keras.Sequential([
            layers.Flatten(input_shape=(self.data.shape[1],)),
            # layers.Dropout(0.3),
            layers.Dropout(0.2),
            layers.Dense(1000, activation='relu'),
            layers.Dense(2),
        ])

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model.fit(self.train_data,
                       validation_data=self.val_data,
                       epochs=self.EPOCHS)


