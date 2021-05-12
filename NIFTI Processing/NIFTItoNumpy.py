import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

np.set_printoptions(threshold=np.inf)

img = nib.load("MRI_PD_29102020.nii")

a = np.array(img.dataobj)


slice1 = a[168, :, :]
# print(slice1)

lst = list(a[170, 160])
print(min(lst))
print(lst)

imgplot = plt.imshow(slice1, cmap="inferno")
plt.show()
