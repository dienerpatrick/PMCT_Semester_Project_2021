import os
import numpy as np
import matplotlib.pyplot as plt

def extract_labels(filedir, labeldir, outdir):

    filenames = os.listdir(filedir)

    ids = []
    cont_labels = []
    bin_labels = []

    for name in filenames:
        ids.append(name[-8:-4])

    label_dict = {}

    with open(labeldir) as labels_file:
        for line in labels_file.readlines()[1:]:
            pair = line.strip("\n").split("\t")
            if len(pair) == 2:
                if pair[1] != '':
                    label_dict[pair[0]] = int(pair[1])

    for fileid in ids:
        cont_labels.append(label_dict[fileid])

    for lab in cont_labels:
        if 0 <= lab <= 50:
            bin_labels.append(0)
        else:
            bin_labels.append(1)

    print(cont_labels)
    print(bin_labels)

    a = cont_labels
    plt.hist(a, bins=40)
    plt.title("histogram")
    plt.show()

    # np.save(outdir + "/cont_labels.npy", np.array(cont_labels))
    # np.save(outdir + "/bin_labels.npy", np.array(bin_labels))



files = os.path.join(os.getcwd(), '..', 'DATA/Sorted_Data/AP_Valid/AP_Valid')
labelfile = os.path.join(os.getcwd(), '..', 'DATA/labels.txt')
out = os.path.join(os.getcwd(), '..', 'DATA/Extracted_Features')

extract_labels(files, labelfile, out)