#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_intrinsic_parameters(image_path):
    # Open image file
    img = Image.open(image_path)

    # Extract EXIF data
    exif_data = img._getexif()

    # Get focal length and sensor dimensions from EXIF
    focal_length = None
    kx = None
    ky = None

    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        # print(tag_name)
        if tag_name == 'FocalLength':
            focal_length = float(value)  # Convert to float
            print(value)
        elif tag_name == 'XResolution':
            kx = float(value)
            print(value)
        elif tag_name == 'YResolution':
            ky = float(value)
            print(value)

    if focal_length is None or kx is None or ky is None:
        raise ValueError("Missing necessary EXIF information")

    # Calculate the intrinsic parameters
    # fx = focal_length * kx
    # fy = focal_length * ky
    # cx = img.width / 2
    # cy = img.height / 2
    fx = focal_length * img.width /6.4
    fy = focal_length * img.height /4.8
    cx = img.width / 2
    cy = img.height / 2

    # Intrinsic parameters matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])

    return intrinsic_matrix

def feature_match(image_a:np.ndarray, image_b: np.ndarray, features:int = 5000, lowe_ratio:float = 0.3, visualization:bool = 0):
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
    points1 = np.float32([imga_key_points[m.queryIdx].pt for m in matches])
    points2 = np.float32([imgb_key_points[m.trainIdx].pt for m in matches])
    return points1, points2

def find_point_matches(image1, image2):
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    print(matches[0])
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # print(matches[])

    # Extract the matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    matches_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
    matches_image = cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('matches image', cv2.WINDOW_NORMAL)
    cv2.imshow('matches image', matches_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points1, points2

def calibrate_cameras(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2):
    # Find the essential matrix
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix1)

    # Recover pose
    _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix1)
    print(R)
    print(t)

    # Create projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # Apply the camera intrinsics
    P1 = cameraMatrix1 @ P1
    P2 = cameraMatrix2 @ P2

    return P1, P2

# Example usage:
img1_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/1.jpg"
img2_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/2.jpg"
# Load images
image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Assuming you already have intrinsic parameters of each camera
# Intrinsic parameters of cameras
intrinsic_matrix1 = get_exif_intrinsic_parameters(img1_path)
intrinsic_matrix2 = get_exif_intrinsic_parameters(img2_path)

# cameraMatrix1 = [...]  # Intrinsic parameters of camera 1
distCoeffs1 = None  # Distortion coefficients of camera 1

# cameraMatrix2 = [...]  # Intrinsic parameters of camera 2
distCoeffs2 = None  # Distortion coefficients of camera 2

points1, points2 = feature_match(image1, image2)
P1, P2 = calibrate_cameras(points1, points2, intrinsic_matrix1, distCoeffs1, intrinsic_matrix2, distCoeffs2)
print(P1)
print('###################################################################3')
print(P2)

