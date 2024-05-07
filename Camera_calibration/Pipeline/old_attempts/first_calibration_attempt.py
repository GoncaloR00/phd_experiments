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
    fx = focal_length * kx / img.width
    fy = focal_length * ky / img.height
    cx = img.width / 2
    cy = img.height / 2

    # Intrinsic parameters matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])

    return intrinsic_matrix


def calibrate_cameras(image_points1, image_points2, intrinsic_matrix1, intrinsic_matrix2):
    # Convert image points to homogeneous coordinates
    image_points1 = np.array(image_points1)
    image_points2 = np.array(image_points2)
    num_points = len(image_points1)

    # Ensure same number of points
    assert len(image_points1) == len(image_points2)

    # Create 3D points (assuming Z=0 in the same plane)
    object_points = np.zeros((num_points, 1, 3), dtype=np.float32)
    object_points[..., :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)  # Assuming a chessboard pattern

    # Calibrate cameras
    retval, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = \
        cv2.stereoCalibrate(object_points, image_points1, image_points2,
                             intrinsic_matrix1, intrinsic_matrix2,
                             imageSize=(640, 480), flags=cv2.CALIB_FIX_INTRINSIC)

    return camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F

# Example usage
if __name__ == "__main__":
    # Load images
    img1_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/1.jpg"
    img2_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/2.jpg"
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Filter good matches
    good_matches = []
    for match in matches:
        if match.distance < 50:  # Adjust this threshold as needed
            good_matches.append(match)
    
    # Get corresponding points
    image_points1 = [kp1[match.queryIdx].pt for match in good_matches]
    image_points2 = [kp2[match.trainIdx].pt for match in good_matches]

    # Intrinsic parameters of cameras
    intrinsic_matrix1 = get_exif_intrinsic_parameters(img1_path)
    intrinsic_matrix2 = get_exif_intrinsic_parameters(img2_path)
    # Calibrate cameras
    camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = \
        calibrate_cameras(image_points1, image_points2, intrinsic_matrix1, intrinsic_matrix2)

    # Print results
    print("Camera 1 Matrix:")
    print(camera_matrix1)
    print("Camera 1 Distortion Coefficients:")
    print(dist_coeffs1)
    print("Camera 2 Matrix:")
    print(camera_matrix2)
    print("Camera 2 Distortion Coefficients:")
    print(dist_coeffs2)
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector:")
    print(T)
    print("Essential Matrix:")
    print(E)
    print("Fundamental Matrix:")
    print(F)
