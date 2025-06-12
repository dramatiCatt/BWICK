import numpy as np
import json
import os
import pathlib
import time
import argparse

import fingerpy as fpa

"""
fingerprint_img_path [--show|-s] [--auth|-a json_file]

eg.: ../Data/Odciski/DB1_B/101_2.tif --show --auth ./Templates/101.json
"""

def test_authentication():
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
            
    for fingerprint_idx, test_fingerprints_paths in enumerate(test_fingerprints_data_paths):
        for test_idx, test_path in enumerate(test_fingerprints_paths):
            test_minutiae = fpa.FingerprintTemplate.from_img_file(test_path).minutiae
            
            for collection_idx, collection in enumerate(fingerprints_templates_collections):
                start = time.perf_counter()

                is_good = fpa.authenticate(test_minutiae, collection, AUTHENTICATION_THRESHOLD)

                end = time.perf_counter()

                print(f"Test {test_idx + 1} of Fingerprint {fingerprint_idx + 1} on Fingerprint {collection_idx + 1}: {'Authenticated' if is_good else 'Not authenticated'} in {end - start:.6f} seconds")

                table_idx = fingerprint_idx * TEST_IMG_PER_FINGERPRINT + test_idx
                if collection_idx == fingerprint_idx:
                    truth_table[table_idx, collection_idx] = True
                else:
                    truth_table[table_idx, collection_idx] = False

                result_table[table_idx, collection_idx] = is_good

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

def main() -> None:
    test_authentication()

    parser = argparse.ArgumentParser(description="Program do porownywania odciskow palcow i testowania autentykacji")
    parser.add_argument("fingerprint_img_path", help="sciezka do zdjecia linij papilarnych palca")
    parser.add_argument("--show", "-s", action="store_true", help="pokazuje wszystkie etapy analizy palca")
    parser.add_argument("--auth", "-a", metavar="templates_json", type=str, help="sciezka do pliku z zapisanymi danymi lini papilarnymi palca")

    args = parser.parse_args()
    
    if not os.path.exists(args.fingerprint_img_path):
        print(f"Blad: Plik '{args.fingerprint_img_path}' nie istnieje.")
        return

    fingerprint = fpa.Fingerprint.from_file(args.fingerprint_img_path)
    if args.show:
        fingerprint.show_all_steps(stop=True)

    template = fingerprint.generate_template()
    
    if args.auth is not None:
        if not os.path.exists(args.auth):
            print(f"Blad: Plik '{args.auth}' nie istnieje.")
            return
        
        templates = fpa.load_templates(args.auth)

        if args.show:
            for saved_template in templates:
                template.show_points(img=fingerprint.skeleton, stop=False)
                saved_template.show_points(stop=True)

        is_good = fpa.authenticate(template.minutiae, templates, 0.4)

        print(f"Autentykacja sie {'' if is_good else 'nie '}powiodla")
    elif args.show:
        template.show_points(img=fingerprint.skeleton, stop=True)

if __name__ == "__main__":
    main()