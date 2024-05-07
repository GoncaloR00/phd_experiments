#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import importlib
import cv2

class txt_color:
   # Class for getting colors for prints
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   ENDBOLD = '\033[22m'

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


# --------------------------------------
# Input arguments
# --------------------------------------

parser = argparse.ArgumentParser(
                    prog='From 2D to 3D',
                    description='This program gets two 2D images with points and the intrinsic parameters of the camera and returns a 3D plot',
                    epilog='Text at the bottom of help')

parser.add_argument('-v', '--verbose',
                    action='store_true')
parser.add_argument('-m', '--matrix',
                    type=str, 
                    help="YAML file containing the camera matrix (intrinsic parameters)",
                    required=True)
parser.add_argument('-i1', '--image1',
                    type=str, 
                    help="Path for the first image", 
                    required=True)
parser.add_argument('-i2', '--image2',
                    type=str, 
                    help="Path for the second image", 
                    required=True)
parser.add_argument('-fm', '--feature_match',
                    type=str, 
                    help="Path for the second image",
                    default="sift", 
                    required=False)
parser.add_argument('-dist', '--distortion',
                    action='store_true',
                    help= "Use if you want to use distortion parameters")

args = parser.parse_args()


# --------------------------------------
# Initialization
# --------------------------------------

print(f"{txt_color.BOLD+txt_color.GREEN}****VERBOSE ON****{txt_color.END}") if args.verbose else 0

# Get image paths
img1_path = args.image1
img2_path = args.image2

# Load images
image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Open the YAML file with the camera parameters
with open(args.matrix, 'r') as file:
    camera_file = yaml.safe_load(file)

# Check if camera_matrix and dist_coeff (if required) are in the YAML file
assert 'camera_matrix' in camera_file, "YAML file missing the 'camera_matrix' key"
assert 'dist_coeff' in camera_file or not(args.distortion), "YAML file missing the 'dist_coeff' key"

# Transform into numpy arrays
camera_matrix = np.array(camera_file['camera_matrix'])
dist_coeff = np.array(camera_file['dist_coeff']) if 'dist_coeff' in camera_file and args.distortion else None
print(f"{txt_color.BOLD+txt_color.GREEN}Loaded camera matrix:{txt_color.ENDBOLD}\n{camera_matrix}{txt_color.END}") if args.verbose else 0
print(f"{txt_color.BOLD+txt_color.GREEN}Loaded distortion coefficients:{txt_color.ENDBOLD}\n{dist_coeff}{txt_color.END}") if args.verbose and args.distortion else 0

# Load the feature match algorithm
fm_algorithm_str = 'Feature_match.' + args.feature_match
fm_algorithm = importlib.import_module(fm_algorithm_str)

points1, points2 = fm_algorithm.feature_match(image1, image2, visualization=True)
P1, P2 = calibrate_cameras(points1, points2, camera_matrix, dist_coeff, camera_matrix, dist_coeff)
triangulate_and_plot(P1, P2, points1, points2)