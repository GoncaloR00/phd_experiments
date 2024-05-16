#!/usr/bin/env python3
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import json

def triangulate_and_plot(P1, P2, points1, points2):
    # Triangulate points
    points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3D = points_hom / points_hom[3]
    
    # Plot 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[0], points_3D[1], points_3D[2])
    plt.show()

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


with open("points1.json", 'r') as f:
    points1 = json.load(f)
with open("points2.json", 'r') as f:
    points2 = json.load(f)


