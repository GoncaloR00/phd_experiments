#!/usr/bin/python3

import cv2
import numpy as np
import ctypes

def args_cpp_nparray(array):
    args = [np.ctypeslib.ndpointer(dtype=array.dtype)]
    for i in range(0, len(array.shape)):
        args.append(ctypes.c_int)
    return args
def cpp_nparray(array):
    return array, *array.shape

# Load the shared library
lib = ctypes.CDLL('./lib_test.so')

points = []
for i in range(0, 1000):
    point = [i, i+5]
    points.append(point)

points = np.array(points, dtype=np.uint16)

# Define the argument types and return type of the function
# lib.process_image.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int]
# lib.process_image.restype = None

# Load an image using OpenCV
image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
image2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")

# Call the C++ function
# lib.process_image(image, image.shape[0], image.shape[1], image.shape[2])


# uint16_t* coord_array, int coord_rows, int coord_cols

print(points)

lib.process_image.argtypes = [*args_cpp_nparray(image), *args_cpp_nparray(image2), *args_cpp_nparray(points)]
lib.process_image(*cpp_nparray(image), *cpp_nparray(image2), *cpp_nparray(points))

print('finhe')



# lib = ctypes.cdll.LoadLibrary('./lib_test.so')
# lib.find_matching_points.argtypes = [np.ctypeslib.ndpointer(dtype=np.uint8)]
    
# img1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
# lib.find_matching_points(img1)
