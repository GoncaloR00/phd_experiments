#!/usr/bin/python3

import numpy as np
import cv2
from random import randint
import copy
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import cProfile
import open3d as o3d
import ctypes
import time


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
    F, mask = cv2.findFundamentalMat(points1, points2)
    # Recover pose
    _, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix1)
    # print(f"Rotation: \n{R} \nTranslation: \n{t}")
    # Create projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    # Apply the camera intrinsics
    P1 = cameraMatrix1 @ P1
    P2 = cameraMatrix2 @ P2
    # return P1, P2
    # print(P1)
    # print('################################')
    # print(P2)
    return R, t, F, P1, P2


def triangulate_and_plot(P1, P2, points1, points2, img1, img2):
    # Triangulate points
    points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3D = points_hom / points_hom[3]
    points_3D_np = np.array(points_3D[:3, :].T, dtype=np.float64)

    # Convert points to integer for indexing
    points1 = points1.astype(int)
    points2 = points2.astype(int)

    # Interpolate colors
    colors1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)[points1[:, 1], points1[:, 0]]
    colors2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)[points2[:, 1], points2[:, 0]]
    # colors = ((colors1 + colors2) / 2).astype(np.uint8)
    colors = colors1.astype(np.uint8)

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D_np)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    
    return pcd
    
def remove_isolated_points(pcd, nb_neighbors=20, std_ratio=2.0):
    # Create a copy of the point cloud
    pcd_clean = pcd

    # Perform statistical outlier removal
    cl, ind = pcd_clean.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Return inlier point cloud
    return cl


def args_cpp_nparray(array):
    args = [np.ctypeslib.ndpointer(dtype=array.dtype)]
    for i in range(0, len(array.shape)):
        args.append(ctypes.c_int)
    return args
def cpp_nparray(array):
    return array, *array.shape


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]

class PointsPair(ctypes.Structure):
    _fields_ = [("first", ctypes.POINTER(Point)), ("first_size", ctypes.c_int), ("second", ctypes.POINTER(Point)), ("second_size", ctypes.c_int)]


def main():
    img1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", cv2.IMREAD_GRAYSCALE)
    img1_back = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
    img2_back = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
    img1_plot = copy.deepcopy(img1_back)
    img2_plot = copy.deepcopy(img2_back)
    K = np.array([[629.400223, 0.000000, 325.240410],
                [0.000000, 627.585852, 262.311140],
                [0.000000, 0.000000, 1.000000]])
    distCoeffs1 = None
    distCoeffs2 = None
    points1, points2 = feature_match(img1, img2, visualization=False)
    R, t, F, P1, P2 = calibrate_cameras(points1, points2, K, distCoeffs1, K, distCoeffs2)
    
    # Save data
    px_values = np.arange(img1.shape[1])
    py_values = np.arange(img1.shape[0])
    py_grid, px_grid = np.meshgrid(py_values, px_values)
    coordinate_array_N = np.dstack([px_grid, py_grid, np.ones_like(px_grid)]).reshape(-1, 3)
    epipolar = np.dot(F, coordinate_array_N.T).T
    print(epipolar[-1])
    coordinate_array = np.dstack([px_grid, py_grid]).reshape(-1, 2).astype(np.uint16)
    points_1 = []
    points_2 = []
    print('Sending to C++')
    time_a = time.time()
    lib = ctypes.CDLL('./libmatching_points.so')
    lib.process_data.restype = PointsPair
    lib.process_data.argtypes = [*args_cpp_nparray(img1_plot), *args_cpp_nparray(img2_plot), *args_cpp_nparray(coordinate_array), *args_cpp_nparray(epipolar), ctypes.c_int, ctypes.c_float]
    data = lib.process_data(*cpp_nparray(img1_plot), *cpp_nparray(img2_plot), *cpp_nparray(coordinate_array), *cpp_nparray(epipolar), 5, 0.8)
    # lib.process_data.argtypes = [*args_cpp_nparray(np.expand_dims(img1, axis=-1)), *args_cpp_nparray(np.expand_dims(img2, axis=-1)), *args_cpp_nparray(coordinate_array), *args_cpp_nparray(epipolar), ctypes.c_int, ctypes.c_float]
    # data = lib.process_data(*cpp_nparray(np.expand_dims(img1, axis=-1)), *cpp_nparray(np.expand_dims(img2, axis=-1)), *cpp_nparray(coordinate_array), *cpp_nparray(epipolar), 5, 0.8)
    for i in range(data.first_size):
        points_1.append([data.first[i].x, data.first[i].y])
    for i in range(data.second_size):
        points_2.append([data.second[i].x, data.second[i].y])
    print(f"Tempo total: {time.time() - time_a}")
    pcd = triangulate_and_plot(P1, P2, np.array(points_1, dtype=float), np.array(points_2, dtype=float), img1_back, img2_back)
    pcd2 = remove_isolated_points(pcd, nb_neighbors=5, std_ratio=0.01)
    o3d.visualization.draw_geometries([pcd2])

if __name__ == "__main__":
    # profile = cProfile.Profile()
    main()
    # profile.print_stats()
    