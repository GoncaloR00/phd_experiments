#!/usr/bin/env python3

from copy import deepcopy
import cv2
import numpy as np

def feature_match(image_a:np.ndarray, image_b: np.ndarray, features:int = 1000, lowe_ratio:float = 0.3, visualization:bool = 0):
    # --------------------------------------
    # Initialization
    # --------------------------------------
    assert 0 < lowe_ratio <=1, "David Lowe's racio must be between 0 and 1"
    sift_detector = cv2.SIFT_create(nfeatures=features)
    # --------------------------------------
    # Execution
    # --------------------------------------

    # Sift features  -----------------------
    imga_key_points, imga_descriptors = sift_detector.detectAndCompute(image_a, None)
    imgb_key_points, imgb_descriptors = sift_detector.detectAndCompute(image_b, None)

    # Match the keypoints
    index_params = dict(algorithm = 1, trees = 15)
    search_params = dict(checks = 50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    two_best_matches = flann_matcher.knnMatch(imgb_descriptors, imga_descriptors, k=2)

    # Create a list of matches
    matches = []
    for match_idx, match in enumerate(two_best_matches):

        best_match = match[0] # to get the cv2.DMatch from the tuple [match = (cv2.DMatch)]
        second_match = match[1]

        # David Lowe's ratio
        if best_match.distance < lowe_ratio * second_match.distance: # this is a robust match, keep it
            matches.append(best_match) # create a list to show with drawMatches


    if visualization:
        matches_image = cv2.drawMatches(image_b, imgb_key_points, image_a, imga_key_points, matches, None)
        matches_image = cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)
        cv2.namedWindow('matches image', cv2.WINDOW_NORMAL)
        cv2.imshow('matches image', matches_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return imga_key_points, imgb_key_points, matches

if __name__ == "__main__":
    image_a = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/1.jpg")
    image_b = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/2.jpg")
    imga_key_points, imgb_key_points, matches = feature_match(image_a, image_b, features =5000, visualization = True)
    # print(f"####################### Image A keypoints #######################\n{imga_key_points}\n####################### Image B keypoints #######################\n{imgb_key_points}\n####################### Matches #######################\n{matches}")