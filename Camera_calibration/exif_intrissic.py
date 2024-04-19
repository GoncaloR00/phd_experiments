#!/usr/bin/env python3
import numpy as np
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
        print(tag_name)
        print(value)
        if tag_name == 'FocalLength':
            focal_length = float(value)  # Convert to float
            # print(value)
        elif tag_name == 'XResolution':
            kx = float(value)
            # print(value)
        elif tag_name == 'YResolution':
            ky = float(value)
            # print(value)

    if focal_length is None or kx is None or ky is None:
        raise ValueError("Missing necessary EXIF information")

    # Calculate the intrinsic parameters
    fx = focal_length * img.width /6.4
    fy = focal_length * img.height /4.8
    cx = img.width / 2
    cy = img.height / 2
    # focal_px = image_width_px * focal_mm / sensor_width_mm
    # Intrinsic parameters matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])

    return intrinsic_matrix

# Example usage
if __name__ == "__main__":
    image_path = "/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/1.jpg"  # Path to your image file
    intrinsic_matrix = get_exif_intrinsic_parameters(image_path)
    print("Intrinsic parameters matrix:")
    print(intrinsic_matrix)
    # try:
    #     intrinsic_matrix = get_exif_intrinsic_parameters(image_path)
    #     print("Intrinsic parameters matrix:")
    #     print(intrinsic_matrix)
    # except Exception as e:
    #     print("Error:", e)