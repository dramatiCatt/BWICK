import numpy as np
import json
import os
import pathlib
import sklearn as skl
import time

import fingerpy as fpa
from timer import print_times, enable_timers, diable_timers

# TODO: Użyć numby lub coś z cudą

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
        template = fpa.create_and_save_fingerprint_templates_collection(fingerprint_paths[TEST_IMG_PER_FINGERPRINT:], 
                                                            f'{templates_folder_path}/1{f"0{f}" if f < 10 else f}.json')
        fingerprints_templates_collections.append(template)

    with open('test_data_paths.json', 'w') as f:
        json.dump(test_fingerprints_data_paths, f, indent=4)
else:
    with open('test_data_paths.json', 'r') as f:
        test_fingerprints_data_paths = json.load(f)
       
    fingerprints_templates_collections = [] 
    templates_folder = pathlib.Path(templates_folder_path)
    for path in [f for f in templates_folder.iterdir() if f.is_file()]:
        fingerprints_templates_collections.append(fpa.load_fingerprint_templates_collection(path))
        

AUTHENTICATION_THRESHOLD = 0.4
MIN_MINUTIAE_TO_AUTHENTICATE = 20

truth_table = np.zeros(shape=(FINGERPRINTS_NUM * TEST_IMG_PER_FINGERPRINT, FINGERPRINTS_NUM), dtype=bool)
result_table = np.zeros_like(truth_table)
        
for fingerprint_idx, test_fingerprints in enumerate(test_fingerprints_data_paths):
    for test_idx, test in enumerate(test_fingerprints):
        test_template = fpa.create_fingerprint_template(test)
        test_minutiae, _, _, _ = fpa.get_data_from_fingerprint_template(test_template)
        
        for collection_idx, collection in enumerate(fingerprints_templates_collections):
            start = time.perf_counter()

            best_match = 0
            for template in collection:
                minutiae, _, _, _ = fpa.get_data_from_fingerprint_template(template)
                matched = fpa.compare_minutiae_sets(test_minutiae, minutiae)
                best_match = max(best_match, matched)

            end = time.perf_counter()

            match_percent = best_match / len(test_template[fpa.TEMPLATE_MINUTIAE])
            print(f"Test {test_idx + 1} of Fingerprint {fingerprint_idx + 1} on Fingerprint {collection_idx + 1}: {best_match} matched minutiae ({match_percent * 100.0} %) matched in {end - start:.6f} seconds")

            if collection_idx == fingerprint_idx:
                truth_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = True
            else:
                truth_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = False

            # if best_match >= MIN_MINUTIAE_TO_AUTHENTICATE:
            #     result_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = True
            #     # print("Autoryzacja pozytywna poprzez minimalne minucje :)))")
            # else:
            #     result_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = False
            #     # print("Autoryzacja negatywna poprzez minimalne minucje :(((")

            if match_percent >= AUTHENTICATION_THRESHOLD:
                result_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = True
                # print("Autoryzacja pozytywna poprzez procenty :)))")
            else:
                result_table[fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx, collection_idx] = False
                # print("Autoryzacja negatywna poprzez procenty :(((")

truth_table = truth_table.flatten()
result_table = result_table.flatten()

print("Classification Stats:")
print(skl.metrics.classification_report(truth_table, result_table, labels=[False, True], digits=4, zero_division=0.0))

positive_num = np.count_nonzero(truth_table)
negative_num = len(truth_table) - positive_num

true_num = np.count_nonzero(result_table)
false_num = len(result_table) - true_num

print(f"True: {true_num}, False: {false_num}, Positive: {positive_num}, Negative: {negative_num}")

# enable_timers()
print_times()