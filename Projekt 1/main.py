import cv2
import numpy as np
import scipy.ndimage as sciimg
import scipy.spatial as scispat
import os
import numpy.random as rnd
import matplotlib.pyplot as plt

import fingerprints_api as fpa

data_folder_path = "../Data/Odciski/DB1_B"

# LOAD FINGERPRINT
fingerprint = fpa.load_img(f"{data_folder_path}/101_2.tif")
rows, columns = fpa.get_img_size(fingerprint)

# fpa.show_img(fingerprint, title="Original")

# NORMALIZE FINGERPRINT
fingerprint_norm = fpa.normalize_img(fingerprint)

# fpa.show_img(fingerprint_norm, title="Normalized")

# CALCULATE ORIENTATION FIELD
# CALCULATE CORSE ORIENTATION FIELD
fingerprint_O, fingerprint_W = fpa.gradient_orientation_field(fingerprint_norm)
# fingerprint_O, fingerprint_W = fpa.average_orientation_field(fingerprint_O, fingerprint_W, block_size=4)

# fpa.show_img(fingerprint_O, title="Fingerprint Orientation", scaleX=16, scaleY=16)
# fpa.show_img(fingerprint_W, title="Fingerprint Weights", scaleX=16, scaleY=16)

plt.figure(figsize=(10, 5))

# POINCARE INDEX
cores_mask, deltas_mask = fpa.poincare_index(fingerprint_O, fingerprint_W, weights_min_power=0.77, close_error=0.1 * np.pi)

print(np.count_nonzero(cores_mask), np.count_nonzero(deltas_mask))

plt.subplot(1, 2, 1)
plt.imshow(fingerprint_O, cmap='gray')
plt.scatter(*np.where(cores_mask)[::-1], color='red', label='Core', s=10)
plt.scatter(*np.where(deltas_mask)[::-1], color='blue', label='Delta', s=10)
plt.legend()
plt.title("Singular Points (Poincare Index)")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.imshow(fingerprint_O, cmap='hsv')
plt.title("Orientation Field")
plt.colorbar()
plt.tight_layout()
plt.axis("equal")

plt.show()

# POLYMONIAL MODEL OF ORIENTATION FIELD
PR, PI = fpa.polymonial_orientation_field(fingerprint_O, fingerprint_W, 4)
fingerprint_PO = 0.5 * np.arctan2(PI, PR)

# POINTS CHARGES
cores_charges = fpa.get_points_charges(fingerprint_O, fingerprint_PO, fingerprint_W, cores_mask, 80)
deltas_charges = fpa.get_points_charges(fingerprint_O, fingerprint_PO, fingerprint_W, deltas_mask, 40)

# POINT CHARGE
final_O = fpa.point_charge_orientation_field(PR, PI, fingerprint_O, cores_mask, deltas_mask, cores_charges, deltas_charges, 80, 40)

plt.figure(figsize=(5, 5))

plt.subplot(1, 1, 1)
plt.imshow(final_O, cmap='gray')
plt.title("Orientation Field")
plt.tight_layout()
plt.axis("equal")
plt.show()

final_O, _ = fpa.average_orientation_field(final_O, fingerprint_W, block_size=16)
fingerprint_O, fingerprint_W = fpa.average_orientation_field(fingerprint_O, fingerprint_W, block_size=16)

overlay = fpa.draw_orientation_field(fingerprint_norm, fingerprint_O, step=16, line_length=14)
cv2.imshow("Orientation Field 1", overlay)

overlay = fpa.draw_orientation_field(fingerprint_norm, final_O, step=16, line_length=14)
cv2.imshow("Orientation Field 2", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: pod liniami by było widać analizowany obraz
# TODO: Zrobić z tego API
# TODO: Użyć numby lub coś z cudą