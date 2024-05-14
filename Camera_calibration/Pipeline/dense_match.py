#!/usr/bin/env python3
import numpy as np
import cv2
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy

def feature_match(image_a:np.ndarray, image_b: np.ndarray, features:int = 5000, lowe_ratio:float = 0.3, visualization:bool = 0):
    assert 0 < lowe_ratio <=1, "David Lowe's racio must be between 0 and 1"
    sift_detector = cv2.SIFT_create(nfeatures=features)
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
    points1 = np.float32([imga_key_points[m.trainIdx].pt for m in matches])
    points2 = np.float32([imgb_key_points[m.queryIdx].pt for m in matches])
    if visualization:
        concatenated_image = np.concatenate((image_a, image_b), axis=1)
        concatenated_image = cv2.cvtColor(concatenated_image,cv2.COLOR_GRAY2RGB)
        for i in range(len(points1)):
            cv2.line(concatenated_image, np.asarray(points1[i], dtype=int), tuple(map(sum, zip(np.asarray(points2[i], dtype=int), (image_a.shape[1], 0)))), (randint(0,255), randint(0,255), randint(0,255)), 1)
        cv2.imshow('Linked Points', concatenated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points1, points2

def calibrate_cameras(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2):
    # Find the essential matrix
    E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix1)
    # Recover pose
    _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix1)
    print(f"Rotation: \n{R} \nTranslation: \n{t}")
    # Create projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    # Apply the camera intrinsics
    P1 = cameraMatrix1 @ P1
    P2 = cameraMatrix2 @ P2
    # return P1, P2
    return R, t

def triangulate_and_plot(P1, P2, points1, points2):
    # Triangulate points
    points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3D = points_hom / points_hom[3]
    
    # Plot 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[0], points_3D[1], points_3D[2])
    plt.show()

def create_dense_map(image1, image2):
    # Convert the images to grayscale
    # gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray1 = copy.deepcopy(image1)
    gray2 = copy.deepcopy(image2)

    # Compute the stereo BM map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray1, gray2)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity = cv2.normalize(disparity, disparity, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)

    # Display the disparity map
    cv2.imshow('Disparity Map', disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disparity

# Example usage:
img1_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png"
img2_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png"

# Load images
image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

height, width= image1.shape

# Intrinsic parameters of cameras
intrinsic_matrix1 = intrinsic_matrix = np.array([[629.400223, 0.000000, 325.240410],
                                                [0.000000, 627.585852, 262.311140],
                                                [0.000000, 0.000000, 1.000000]])
intrinsic_matrix2 = intrinsic_matrix = np.array([[629.400223, 0.000000, 325.240410],
                                                [0.000000, 627.585852, 262.311140],
                                                [0.000000, 0.000000, 1.000000]])
distCoeffs1 = None  # Distortion coefficients of camera 1
distCoeffs2 = None  # Distortion coefficients of camera 2

points1, points2 = feature_match(image1, image2, visualization=True)
# P1, P2 = calibrate_cameras(points1, points2, intrinsic_matrix1, distCoeffs1, intrinsic_matrix2, distCoeffs2)
# triangulate_and_plot(P1, P2, points1, points2)
R, T = calibrate_cameras(points1, points2, intrinsic_matrix1, distCoeffs1, intrinsic_matrix2, distCoeffs2)

# Stereo rectification
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(intrinsic_matrix1, distCoeffs1, intrinsic_matrix2, distCoeffs2, (width, height), R, T)

# Now Q is the disparity-to-depth mapping matrix



# Example usage:
disparity_map = create_dense_map(image1, image2)

# Now you can use the disparity map to create a 3D plot
points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
mask_map = disparity_map > disparity_map.min()
output_points = points_3D[mask_map]

# Create a color map
colors = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
output_colors = colors[mask_map]

# Define the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Populate the figure with our data
ax.scatter(output_points[:, 0], output_points[:, 1], output_points[:, 2], c=output_colors/255, s=3)
plt.show()