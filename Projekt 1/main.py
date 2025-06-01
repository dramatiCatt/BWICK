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
fingerprint_O, fingerprint_W = fpa.average_orientation_field(fingerprint_O, fingerprint_W, block_size=8)

# fpa.show_img(fingerprint_O, title="Fingerprint Orientation", scaleX=16, scaleY=16)
# fpa.show_img(fingerprint_W, title="Fingerprint Weights", scaleX=16, scaleY=16)

cos2O = np.cos(2 * fingerprint_O)
sin2O = np.sin(2 * fingerprint_O)

# cv2.imshow("Cos(2O)", cv2.resize(cos2O, None, fx=1, fy=1))
# cv2.imshow("Sin(2O)", cv2.resize(sin2O, None, fx=1, fy=1))

plt.figure(figsize=(20, 5))

x, y = np.meshgrid(np.arange(columns), np.arange(rows))

# plt.subplot(1, 4, 1)
# plt.title("Orginalne pole orientacji")
# plt.quiver(x, y, cos2O, sin2O, fingerprint_O)
# plt.gca().invert_yaxis()
# plt.axis("equal")

plt.tight_layout()

# POINCARE INDEX
cores_mask, deltas_mask = fpa.poincare_index(fingerprint_O, fingerprint_W, weights_min_power=0.2, close_error=0.5 * np.pi)

print(np.count_nonzero(cores_mask), np.count_nonzero(deltas_mask))

plt.subplot(1, 4, 2)
plt.imshow(fingerprint_O, cmap='gray')
plt.scatter(*np.where(cores_mask)[::-1], color='red', label='Core', s=10)
plt.scatter(*np.where(deltas_mask)[::-1], color='blue', label='Delta', s=10)
plt.legend()
plt.title("Singular Points (Poincare Index)")
plt.tight_layout()
plt.axis("equal")

plt.subplot(1, 4, 4)
plt.imshow(fingerprint_O, cmap='hsv')
plt.title("Orientation Field")
plt.colorbar()
plt.tight_layout()
plt.axis("equal")

plt.show()

overlay = fpa.draw_orientation_field(fingerprint_norm, fingerprint_O, step=8, line_length=6)
cv2.imshow("Orientation Field", overlay)

# POINT CHARGE
final_O = fpa.point_charge_orientation_field(fingerprint_O, cores_mask, deltas_mask)

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 3)
plt.imshow(final_O, cmap='gray')
plt.title("Orientation Field")
plt.tight_layout()
plt.axis("equal")

overlay = fpa.draw_orientation_field(fingerprint_norm, final_O, step=8, line_length=6)
cv2.imshow("Orientation Field", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO: pod liniami by było widać analizowany obraz
# TODO: Zrobić z tego API
# TODO: Użyć numby lub coś z cudą