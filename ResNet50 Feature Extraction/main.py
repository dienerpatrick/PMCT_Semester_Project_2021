from DenseTop import DenseModel
import numpy as np
import os

DATADIR = os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features')

ap_data = np.load(os.path.join(DATADIR, "ap_features224.npy"))
l_data = np.load(os.path.join(DATADIR, "l_features224.npy"))
bin_labels = np.load(os.path.join(DATADIR, "bin_labels.npy"))

dense_model = DenseModel(ap_data, l_data, bin_labels)

