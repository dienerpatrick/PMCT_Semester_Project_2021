import numpy as np
import tensorflow as tf
import os


# arr_ap = np.load(os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features/ap_features.npy'))
# arr_l = np.load(os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features/l_features.npy'))
# labels = np.load(os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features/labels.npy'))

# arr1 = np.array([[1, 2], [3, 4], [5, 6]])
# arr2 = np.array([[7, 8], [9, 10], [11, 12]])
#
# print(np.concatenate((arr1, arr2), axis=1))
# print(arr1.shape)
# print(np.concatenate((arr1, arr2), axis=1).shape)


DATADIR = os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features')

ap_data = np.load(os.path.join(DATADIR, "ap_features224.npy"))
l_data = np.load(os.path.join(DATADIR, "l_features224.npy"))
bin_labels = np.load(os.path.join(DATADIR, "bin_labels.npy"))

conc = np.concatenate((ap_data, l_data), axis=1)

print(ap_data.shape)
print(conc.shape)
print(bin_labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((conc, bin_labels))
print(dataset)