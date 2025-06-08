import numpy as np

import fingerprints_api as fpa

# TODO: Użyć numby lub coś z cudą

data_folder_path = "../Data/Odciski/DB1_B"
templates_folder_path = './Templates'

# CREATE DATASET
IMG_PER_FINGERPRINT = 8
FINGERPRINTS_NUM = 10

fingerprints_data_paths = []
for f in range(1, FINGERPRINTS_NUM + 1):
    fingerprint_paths = []
    for i in range(1, IMG_PER_FINGERPRINT + 1):
        fingerprint_paths.append(f"{data_folder_path}/1{f'0{f}' if f < 10 else f}_{i}.tif")

    # RANDOMIZE PATHS PER FINGERPRINT
    np.random.shuffle(fingerprint_paths)
    fingerprints_data_paths.append(fingerprint_paths)

# GET TEST DATASET AND TEMPLATES
TEST_DATA_PERCENT = 0.3


# LOAD FINGERPRINT
fingerprint = fpa.load_img(f"{data_folder_path}/101_2.tif")

minutiae, core_point, delta_point, core_angle = fpa.get_fingerprint_data(fingerprint, 16, 5, 5, 0.5, 0.4 * np.pi)

fpa.save_fingerprint_template(f'{templates_folder_path}/temp.json', minutiae, core_point, delta_point, core_angle)

minutiae, core_point, delta_point, core_angle = fpa.load_fingerprint_template(f'{templates_folder_path}/temp.json')