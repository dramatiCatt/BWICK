import cv2
import numpy as np
import os

data_folder_path = "../Data/Odciski/DB1_B"

fingerprint_test = cv2.imread(f"{data_folder_path}/101_2.tif")
cv2.imshow("Original", cv2.resize(fingerprint_test, None, fx=1, fy=1))
cv2.waitKey(0)
cv2.destroyAllWindows()

for file in [file for file in os.listdir(data_folder_path)]:
    fingerprint_database_image = cv2.imread(f"{data_folder_path}/{file}")
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_test, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), 
                dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
    
    for p, q in matches:
        if p.distance < 0.1*q.distance:
            match_points.append(p)
    
    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)            
    else:
        keypoints = len(keypoints_2)

    if (len(match_points) / keypoints)>0.95:
        print("% match: ", len(match_points) / keypoints * 100)
        print("Figerprint ID: " + str(file)) 
        result = cv2.drawMatches(fingerprint_test, keypoints_1, fingerprint_database_image, 
                                keypoints_2, match_points, None) 
        result = cv2.resize(result, None, fx=2.5, fy=2.5)
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break;