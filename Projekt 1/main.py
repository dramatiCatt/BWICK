import numpy as np
import json
import os
import pathlib
import sklearn as skl
import time

import fingerpy as fpa
from timer import print_times, enable_timers

# enable_timers()

data_folder_path = "../Data/Odciski/DB1_B"
templates_folder_path = './Templates'

# CREATE DATASET
FORCE_CREATE_DATASET = False

IMG_PER_FINGERPRINT = 8
FINGERPRINTS_NUM = 10

TEST_DATA_PERCENT = 0.3
TEST_IMG_PER_FINGERPRINT = int(TEST_DATA_PERCENT * IMG_PER_FINGERPRINT)

if not os.path.exists('test_data_paths.json') or FORCE_CREATE_DATASET:
    test_fingerprints_data_paths = []
    fingerprints_templates_collections = []
    for f in range(1, FINGERPRINTS_NUM + 1):
        fingerprint_paths = []
        for i in range(1, IMG_PER_FINGERPRINT + 1):
            fingerprint_paths.append(f"{data_folder_path}/1{f'0{f}' if f < 10 else f}_{i}.tif")

        # RANDOMIZE PATHS PER FINGERPRINT
        np.random.shuffle(fingerprint_paths)

        # GET TEST DATASET AND TEMPLATES
        test_fingerprints_data_paths.append(fingerprint_paths[:TEST_IMG_PER_FINGERPRINT])
        
        # SAVE TEMPLATES
        template = fpa.create_and_save_templates(f'{templates_folder_path}/1{f"0{f}" if f < 10 else f}.json', 
                                      fingerprint_paths[TEST_IMG_PER_FINGERPRINT:])
        fingerprints_templates_collections.append(template)

    with open('test_data_paths.json', 'w') as f:
        json.dump(test_fingerprints_data_paths, f, indent=4)
else:
    with open('test_data_paths.json', 'r') as f:
        test_fingerprints_data_paths = json.load(f)
       
    fingerprints_templates_collections = [] 
    templates_folder = pathlib.Path(templates_folder_path)
    for path in [f for f in templates_folder.iterdir() if f.is_file()]:
        fingerprints_templates_collections.append(fpa.load_templates(path))
        

AUTHENTICATION_THRESHOLD = 0.4

truth_table = np.zeros(shape=(FINGERPRINTS_NUM * TEST_IMG_PER_FINGERPRINT, FINGERPRINTS_NUM), dtype=bool)
result_table = np.zeros_like(truth_table)
        
for fingerprint_idx, test_fingerprints in enumerate(test_fingerprints_data_paths):
    for test_idx, test in enumerate(test_fingerprints):
        test_template = fpa.FingerprintTemplate.from_img_file(test)
        test_minutiae = test_template.minutiae
        
        for collection_idx, collection in enumerate(fingerprints_templates_collections):
            start = time.perf_counter()

            best_match = 0
            for template in collection:
                minutiae = template.minutiae
                matched = fpa.minuti.compare_minutiae_sets(test_minutiae, minutiae)
                best_match = max(best_match, matched)

            end = time.perf_counter()

            match_percent = best_match / len(test_minutiae)
            print(f"Test {test_idx + 1} of Fingerprint {fingerprint_idx + 1} on Fingerprint {collection_idx + 1}: {best_match} matched minutiae ({match_percent * 100.0} %) matched in {end - start:.6f} seconds")

            if collection_idx == fingerprint_idx:
                truth_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = True
            else:
                truth_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = False

            if match_percent >= AUTHENTICATION_THRESHOLD:
                result_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = True
            else:
                result_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = False

truth_table = truth_table.flatten()
result_table = result_table.flatten()

# print("Classification Stats:")
# print(skl.metrics.classification_report(truth_table, result_table, labels=[False, True], digits=4, zero_division=0.0))

true_false_table = truth_table == result_table
true_mask = np.argwhere(true_false_table)

true_table = result_table[true_mask]
true_positive_num = np.count_nonzero(true_table)
true_negative_num = len(true_table) - true_positive_num

false_positive_num = np.count_nonzero(result_table) - true_positive_num
false_negative_num = len(result_table) - false_positive_num - true_positive_num - true_negative_num

print()
print(f"Results num: {len(result_table)}")
print(f"True Positive: {true_positive_num}, expected: {FINGERPRINTS_NUM * TEST_IMG_PER_FINGERPRINT}")
print(f"True Negative: {true_negative_num}, expected: {(FINGERPRINTS_NUM ** 2 - FINGERPRINTS_NUM) * TEST_IMG_PER_FINGERPRINT}")
print(f"False Positive: {false_positive_num}, expected: 0")
print(f"False Negative: {false_negative_num}, expected: 0")

# print_times()