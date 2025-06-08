import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as markers

import fingerprints_api as fpa

# TODO: Użyć numby lub coś z cudą

data_folder_path = "../Data/Odciski/DB1_B"

# LOAD FINGERPRINT
fingerprint = fpa.load_img(f"{data_folder_path}/101_2.tif")

binarized, skeleton, contour, border_contour, orientation_field, reliability_map, core_point, delta_point = fpa.get_fingerprint_data(fingerprint, 16, 5, 5, 0.5, 0.4 * np.pi, 40)

minutae = fpa.extract_minutiae(skeleton, reliability_map, orientation_field, 16, 0.5)

endings_mask = [[m['x'], m['y']] for m in minutae if m['type'] == fpa.MINUTIAE_ENDING]
bifurcation_mask = [[m['x'], m['y']] for m in minutae if m['type'] == fpa.MINUTIAE_BIFURCATION]

small_orientation_field, small_weight = fpa.average_orientation_field(orientation_field, reliability_map, block_size=8)

plt.figure(figsize=(5, 5))

plt.imshow(skeleton, cmap='gray')
if core_point is not None:
    plt.scatter(core_point[1], core_point[0], color='blue', label='Core', s=10, marker='+')
if delta_point is not None:
    plt.scatter(delta_point[1], delta_point[0], color='red', label='Delta', s=10, marker='.')
plt.scatter(np.array(endings_mask)[..., 0], np.array(endings_mask)[..., 1], color='green', label='Ending', s=10, marker=markers.TICKLEFT)
plt.scatter(np.array(bifurcation_mask)[..., 0], np.array(bifurcation_mask)[..., 1], color='violet', label='Bifurcation', s=10,  marker='1')
plt.legend()
plt.title("Singular Points (Poincare Index) And Minuates")
plt.tight_layout()
plt.axis("equal")
plt.show()

# fpa.show_img(binarized, "Binarized")

# fpa.show_img(skeleton, "Skeleton")

# fpa.show_img(reliability_map, "Reliability")

# overlay = fpa.draw_orientation_field(None, small_orientation_field, small_weight, step=8, line_length=6)
# fpa.show_img(overlay, title="Orientation Field 1")

cv2.waitKey(0)
cv2.destroyAllWindows()