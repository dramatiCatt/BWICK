import cv2
import numpy as np
import scipy.ndimage as sciimg
import scipy.spatial as scispat
import os
import numpy.random as rnd
import matplotlib.pyplot as plt
import sklearn as skl

import fingerprints_api as fpa

# TODO: Użyć numby lub coś z cudą

data_folder_path = "../Data/Odciski/DB1_B"

# LOAD FINGERPRINT
fingerprint = fpa.load_img(f"{data_folder_path}/101_2.tif")

binarized, skeleton, orientation_field, weight, cores_mask, deltas_mask = fpa.get_fingerprint_data(fingerprint, 16, 5, 5, 0.68, 0)

print(np.count_nonzero(cores_mask), np.count_nonzero(deltas_mask))

small_orientation_field, small_weight = fpa.average_orientation_field(orientation_field, weight, block_size=8)

plt.figure(figsize=(5, 5))

plt.imshow(orientation_field, cmap='gray')
plt.scatter(*np.where(cores_mask)[::-1], color='red', label='Core', s=10)
plt.scatter(*np.where(deltas_mask)[::-1], color='blue', label='Delta', s=10)
plt.legend()
plt.title("Singular Points (Poincare Index)")
plt.tight_layout()
plt.axis("equal")
plt.show()

fpa.show_img(binarized, "Binarized")

fpa.show_img(skeleton, "Skeleton")

fpa.show_img(weight, "Weights")

overlay = fpa.draw_orientation_field(binarized, small_orientation_field, small_weight, step=8, line_length=6)
fpa.show_img(overlay, title="Orientation Field 1")

cv2.waitKey(0)
cv2.destroyAllWindows()