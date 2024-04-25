#!/usr/bin/env python3
import numpy as np
import cv2
from random import randint

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
    # if visualization:
    #     matches_image = cv2.drawMatches(image_b, imgb_key_points, image_a, imga_key_points, matches, None)
    #     matches_image = cv2.cvtColor(matches_image, cv2.COLOR_BGR2RGB)
    #     cv2.namedWindow('matches image', cv2.WINDOW_NORMAL)
    #     cv2.imshow('matches image', matches_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # print(len(imga_key_points))
    # print(matches[288].queryIdx)
    # print(len(imgb_key_points))
    # print(matches[288].trainIdx)
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

    return P1, P2
def calibrate_camerasv2(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2):
    # Undistort points
    points1 = cv2.undistortPoints(points1, cameraMatrix1, distCoeffs1)
    points2 = cv2.undistortPoints(points2, cameraMatrix2, distCoeffs2)

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

    return P1, P2, points1, points2
def calibrate_camerasv3(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2):
    # Undistort points
    points1_undist = cv2.undistortPoints(points1, cameraMatrix1, distCoeffs1)
    points2_undist = cv2.undistortPoints(points2, cameraMatrix2, distCoeffs2)

    # Convert back to pixel coordinates
    points1 = cv2.convertPointsFromHomogeneous((cameraMatrix1 @ np.concatenate((points1_undist, np.ones((points1_undist.shape[0], 1, 1))), axis=-1).transpose(0, 2, 1)).transpose(0, 2, 1))
    points2 = cv2.convertPointsFromHomogeneous((cameraMatrix2 @ np.concatenate((points2_undist, np.ones((points2_undist.shape[0], 1, 1))), axis=-1).transpose(0, 2, 1)).transpose(0, 2, 1))

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

    return P1, P2, points1, points2



# def triangulate_points(P1, P2, points1, points2):
#     # Convert points to homogeneous coordinates
#     points1_h = cv2.convertPointsToHomogeneous(np.asarray(points1, dtype=int))
#     points2_h = cv2.convertPointsToHomogeneous(np.asarray(points2, dtype=int))
#     print(points1_h)
#     # Triangulate points
#     points_4D = cv2.triangulatePoints(P1, P2, points1_h, points2_h)

#     # Convert to non-homogeneous coordinates
#     points_3D = cv2.convertPointsFromHomogeneous(points_4D.T)

#     return points_3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def triangulate_and_plot(P1, P2, points1, points2):
    # Triangulate points
    points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3D = points_hom / points_hom[3]
    
    # Plot 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[0], points_3D[1], points_3D[2])
    plt.show()

# Example usage:
img1_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/img0_0.png"
img2_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/img0_1.png"
# Load images
image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Assuming you already have intrinsic parameters of each camera
# Intrinsic parameters of cameras
intrinsic_matrix1 = intrinsic_matrix = np.array([[707.717612, 0, 304.802554],
                                                [0, 707.680471, 227.285062],
                                                [0, 0, 1]])
intrinsic_matrix2 = intrinsic_matrix = np.array([[707.717612, 0, 304.802554],
                                                [0, 707.680471, 227.285062],
                                                [0, 0, 1]])

# cameraMatrix1 = [...]  # Intrinsic parameters of camera 1
distCoeffs1 = None  # Distortion coefficients of camera 1
# distCoeffs1 = np.array([0.228289, -0.244408, -0.002493, -0.012352, 0])

# cameraMatrix2 = [...]  # Intrinsic parameters of camera 2
distCoeffs2 = None  # Distortion coefficients of camera 2
# distCoeffs2 = np.array([0.228289, -0.244408, -0.002493, -0.012352, 0])

points1, points2 = feature_match(image1, image2)
# points1 = cv2.undistortPoints(np.array(points1), intrinsic_matrix1, distCoeffs1)
# points2 = cv2.undistortPoints(np.array(points2), intrinsic_matrix2, distCoeffs2)
P1, P2 = calibrate_cameras(points1, points2, intrinsic_matrix1, distCoeffs1, intrinsic_matrix2, distCoeffs2)
triangulate_and_plot(P1, P2, points1, points2)
P1, P2, points1, points2 = calibrate_camerasv3(points1, points2, intrinsic_matrix1, distCoeffs1, intrinsic_matrix2, distCoeffs2)

triangulate_and_plot(P1, P2, points1, points2)

# triangulate_and_plot(P1, P2, points1_normalized, points2_normalized)

# points3d = triangulate_points(P1, P2, points1_normalized, points2_normalized)
# print(points3d)
# print(P1)
# print('###################################################################3')
# print(P2)


