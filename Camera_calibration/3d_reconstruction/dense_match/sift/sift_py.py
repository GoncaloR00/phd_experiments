#!/usr/bin/python3


import numpy as np
import cv2
from random import randint
# import imutils
import math


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def rotation(img1, img2, coord, epipolar_line, window_size):
    a, b, c = epipolar_line
    rotation = math.atan((a/b))*(180/math.pi)
    rotated_image = rotate_image(img1, rotation)
    cv2.imshow('teste', rotated_image)
    cv2.waitKey(0)



# def feature_match(img1:np.ndarray, img2: np.ndarray, features:int = 5000, lowe_ratio:float = 0.3, visualization:bool = 0):
#     assert 0 < lowe_ratio <=1, "David Lowe's racio must be between 0 and 1"
#     img1_mono = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2_mono = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#     # TODO Discover why the results are different converting to grayscale with imread
    
#     # img1_mono = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", cv2.IMREAD_GRAYSCALE)
#     # img2_mono = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", cv2.IMREAD_GRAYSCALE)
#     sift_detector = cv2.SIFT_create(nfeatures=features)
#     # Sift features  -----------------------
#     imga_key_points, imga_descriptors = sift_detector.detectAndCompute(img1_mono, None)
#     imgb_key_points, imgb_descriptors = sift_detector.detectAndCompute(img2_mono, None)
#     # Match the keypoints
#     index_params = dict(algorithm = 1, trees = 15)
#     search_params = dict(checks = 50)
#     flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
#     two_best_matches = flann_matcher.knnMatch(imgb_descriptors, imga_descriptors, k=2)
#     # Create a list of matches
#     matches = []
#     for match_idx, match in enumerate(two_best_matches):
#         best_match = match[0] # to get the cv2.DMatch from the tuple [match = (cv2.DMatch)]
#         second_match = match[1]
#         # David Lowe's ratio
#         if best_match.distance < lowe_ratio * second_match.distance: # this is a robust match, keep it
#             matches.append(best_match) # create a list to show with drawMatches
#     points1 = np.float32([imga_key_points[m.trainIdx].pt for m in matches])
#     points2 = np.float32([imgb_key_points[m.queryIdx].pt for m in matches])
#     if visualization:
#         concatenated_image = np.concatenate((img1_mono, img2_mono), axis=1)
#         concatenated_image = cv2.cvtColor(concatenated_image,cv2.COLOR_GRAY2RGB)
#         for i in range(len(points1)):
#             cv2.line(concatenated_image, np.asarray(points1[i], dtype=int), tuple(map(sum, zip(np.asarray(points2[i], dtype=int), (img1_mono.shape[1], 0)))), (randint(0,255), randint(0,255), randint(0,255)), 1)
#         cv2.imshow('Linked Points', concatenated_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     return points1, points2




img1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
img2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
coord = None
epipolar_line = (1,1,10)
window_size = 10
rotation(img1, img2, coord, epipolar_line, window_size)