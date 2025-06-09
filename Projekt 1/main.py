import numpy as np
import json
import os
import pathlib

import fingerprints_api as fpa

# TODO: Użyć numby lub coś z cudą

data_folder_path = "../Data/Odciski/DB1_B"
templates_folder_path = './Templates'

# CREATE DATASET
FORCE_CREATE_DATASET = True

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
        

AUTHENTICATION_THRESHOLD = 0.7
        
for fingerprint_idx, test_fingerprints in enumerate(test_fingerprints_data_paths):
    for test_idx, test in enumerate(test_fingerprints):
        test_template = fpa.create_fingerprint_template(test)
        test_minutiae, _, _, _ = fpa.get_data_from_fingerprint_template(test_template)
        
        best_match = 0
        for template in fingerprints_templates_collections[fingerprint_idx]:
            minutiae, _, _, _ = fpa.get_data_from_fingerprint_template(template)
            matched = fpa.compare_minutiae_sets(test_minutiae, minutiae)
            best_match = max(best_match, matched)
        
        match_percent = best_match / len(test_template[fpa.TEMPLATE_MINUTIAE])
        print(f"Fingerprint {fingerprint_idx + 1} Test {test_idx + 1}: {best_match} matched minutiae ({match_percent * 100.0} %)")
        if match_percent >= AUTHENTICATION_THRESHOLD:
            print("Autoryzacja pozytywna :)))")
        else:
            print("Autoryzacja negatywna :(((")