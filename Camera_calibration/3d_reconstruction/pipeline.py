#!/usr/bin/python3

import sys
sys.dont_write_bytecode = True

import numpy as np
import cv2
from epipolar_lines.epipolar_lines_py import epipolar_lines
from feature_match.feature_match import feature_match
from calibration.calibration import calibrate_cameras
from triangulate.triangulate import triangulate
from dense_match.dense_match import dense_match
from outlier_removal.outlier_removal import outlier_removal

import open3d as o3d




def main():
    print('Loading Data...')
    img1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
    img2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")

    K = np.array([[629.400223, 0.000000, 325.240410],
                [0.000000, 627.585852, 262.311140],
                [0.000000, 0.000000, 1.000000]])
    distCoeffs1 = None
    distCoeffs2 = None
    print('Feature matching (calibration)...')
    points1, points2 = feature_match(img1, img2, visualization=False)
    print('Calibration...')
    R, t, F, P1, P2 = calibrate_cameras(points1, points2, K, distCoeffs1, K, distCoeffs2)
    print('Computing and organizing data...')
    epipolar_array, coordinate_array, _ = epipolar_lines(img1, F)
    print('Dense matching...')
    points_1, points_2 = dense_match(img1, img2, coordinate_array, epipolar_array, algorithm='norm_l1')
    print('Generating point cloud...')
    pcd = triangulate(P1, P2, np.array(points_1, dtype=float), np.array(points_2, dtype=float), img1, img2)
    print('Removing outliers...')
    pcd = outlier_removal(pcd, nb_neighbors=1000, std_ratio=0.1)
    main_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
    second_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t.reshape(-1)
    second_frame.transform(transformation)
    o3d.visualization.draw_geometries([pcd, main_frame, second_frame])

if __name__ == "__main__":
    main()