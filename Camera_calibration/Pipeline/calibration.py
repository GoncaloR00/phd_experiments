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

def find_point_matches(image1, image2):
    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Match descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return points1, points2

def calibrate_cameras(points1, points2, intrinsic_matrix1, intrinsic_matrix2):
    # Calibrate cameras using point correspondences and known intrinsic parameters
    _, rotation_matrix, translation_vector, _, _ = cv2.stereoCalibrate(
        objectPoints=np.zeros((1, 1, 3), dtype=np.float32),  # Dummy object points
        imagePoints1=points1,
        imagePoints2=points2,
        cameraMatrix1=intrinsic_matrix1,
        distCoeffs1=None,
        cameraMatrix2=intrinsic_matrix2,
        distCoeffs2=None,
        imageSize=(image1.shape[1], image1.shape[0]),  # Assuming both images have the same size
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    return rotation_matrix, translation_vector

# Example usage
if __name__ == "__main__":
    img1_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/1.jpg"
    img2_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/2.jpg"
    # Load images
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Assuming you already have intrinsic parameters of each camera
    # Intrinsic parameters of cameras
    intrinsic_matrix1 = get_exif_intrinsic_parameters(img1_path)
    intrinsic_matrix2 = get_exif_intrinsic_parameters(img2_path)

    # Find point matches between images
    points1, points2 = find_point_matches(image1, image2)

    # Calibrate cameras
    rotation_matrix, translation_vector = calibrate_cameras(points1, points2, intrinsic_matrix1, intrinsic_matrix2)

    print("Rotation Matrix:")
    print(rotation_matrix)
    print("\nTranslation Vector:")
    print(translation_vector)
