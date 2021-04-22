import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os


class FeatureExtract:
    def __init__(self):

        self.sorted_data_dir = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data')

        self.ap_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.sorted_data_dir + "/AP_Valid",
            image_size=(224, 224),
            color_mode='rgb',
            batch_size=32)

        self.l_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.sorted_data_dir + "/L_Valid",
            image_size=(224, 224),
            color_mode='rgb',
            batch_size=32)

        self.base_model = ResNet50(input_shape=(224, 224, 3),
                                   include_top=False,
                                   weights="imagenet")

        self.ap_features = self.base_model.predict(self.ap_data)
        print(self.ap_features.shape)
        self.ap_features = self.ap_features.reshape((self.ap_features.shape[0], 7 * 7 * 2048))

        self.l_features = self.base_model.predict(self.l_data)
        self.l_features = self.ap_features.reshape((self.l_features.shape[0], 7 * 7 * 2048))

        np.save(os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features/ap_features224.npy'), self.ap_features)
        np.save(os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features/l_features224.npy'), self.l_features)


features = FeatureExtract()
