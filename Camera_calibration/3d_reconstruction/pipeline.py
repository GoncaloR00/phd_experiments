#!/usr/bin/python3

import sys
sys.dont_write_bytecode = True

import numpy as np
import cv2
import argparse
from epipolar_lines.epipolar_lines_py import epipolar_lines
from feature_match.feature_match import feature_match
from calibration.calibration import calibrate_cameras
from triangulate.triangulate import triangulate
from dense_match.dense_match import dense_match
from outlier_removal.outlier_removal import outlier_removal

import open3d as o3d


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rotation_mat

def calculate_new_coordinates(original_x, original_y, M):
    original_point = np.array([[original_x, original_y]])
    new_point = cv2.transform(original_point.reshape(-1, 1, 2), M).reshape(2)
    return int(round(new_point[0])), int(round(new_point[1]))

def reverse_coords(new_x, new_y, M):
    inverse = np.linalg.inv(M)
    print(inverse)
    new_point = np.array([[new_x, new_y]])
    original_point = cv2.transform(new_point.reshape(-1, 1, 2), inverse).reshape(2)
    return int(round(original_point[0])), int(round(original_point[1]))

import math
# def tempi(epipolar_line, img2, window_size):
#     window_size += 1 if window_size % 2 == 0 else 0 # Ensure odd number to get the pixel in the center of the window
#     a, b, c = epipolar_line
#     k = (a/b)+1
#     print(k)
#     pre_window_size = math.ceil(k*window_size)
#     pre_window_size += 1 if pre_window_size % 2 == 0 else 0 # Ensure odd number to get the pixel in the center of the window
#     rotation = -math.atan((a/b))*(180/math.pi)
#     rows, cols = img2.shape[:2]
#     x0, y0 = map(int, [0, -c/b])
#     x1, y1 = map(int, [cols, -(c + a*cols) / b])
#     point = (int((x1+x0)/2), int((y1+y0)/2))
#     img2 = cv2.line(img2, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=1)
#     c_old = c
#     c= c_old + b*pre_window_size/2
#     x0, y0 = map(int, [0, -c/b])
#     x1, y1 = map(int, [cols, -(c + a*cols) / b])
#     img2 = cv2.line(img2, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=1)
#     c= c_old - b*pre_window_size/2
#     x0, y0 = map(int, [0, -c/b])
#     x1, y1 = map(int, [cols, -(c + a*cols) / b])
#     img2 = cv2.line(img2, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=1)

#     img2 = cv2.rectangle(img2, (point[0]-int(pre_window_size/2),point[1]-int(pre_window_size/2)),(point[0]+int(pre_window_size/2),point[1]+int(pre_window_size/2)), color=(255, 0, 0), thickness=1)
    
#     rotated_image, M = rotate_image(img2, rotation)


#     img2 = cv2.circle(img2, point, 1, (0,0,255), 2)
#     # new_point = calculate_new_coordinates(0, 0, cols, rows, rotation)
#     new_point = calculate_new_coordinates(point[0], point[1], M)

#     rotated_image = cv2.circle(rotated_image, new_point, 1, (0,0,255), 2)
#     rotated_image = cv2.rectangle(rotated_image, (new_point[0]-int(window_size/2),new_point[1]-int(window_size/2)),(new_point[0]+int(window_size/2),new_point[1]+int(window_size/2)), color=(255, 0, 0), thickness=1)
#     cv2.imshow('teste', rotated_image)
#     cv2.imshow('original', img2)
#     cv2.waitKey(0)

def tempi(epipolar_line, img2, window_size, up:bool):
    window_size += 1 if window_size % 2 == 0 else 0 # Ensure odd number to get the pixel in the center of the window
    a, b, c = epipolar_line
    rotation = -math.atan((a/b))*(180/math.pi)
    rows, cols = img2.shape[:2]
    x0, y0 = map(int, [0, -c/b])
    x1, y1 = map(int, [cols, -(c + a*cols) / b])




    point = (int((x1+x0)/2), int((y1+y0)/2))


    img2 = cv2.line(img2, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=1)

    rotated_image, M = rotate_image(img2, rotation)

    _ , new_y0 = calculate_new_coordinates(0, -c/b, M)
    # M = np.append(M, [[0, 0, 0]])
    M2 = np.vstack([M, [0, 0, 0]])
    print(M)
    # Quando uma linha na diagonal é colocada na horizontal, a sua espessura passa a ser de 2 pixels em vez de 1 pixel. É preciso escolher se a analise é feita no pixel de cima ou no de baixo: Qual deverá ser o critério para selecionar?
    y0_low = new_y0 - int(window_size/2) -1 if up else new_y0 - int(window_size/2)
    y0_high = new_y0 + int(window_size/2) if up else new_y0 + int(window_size/2)+1


    img2 = cv2.circle(img2, point, 1, (0,0,255), 2)

    new_point = calculate_new_coordinates(point[0], point[1], M)
    


    try_point = reverse_coords(new_point[0], new_point[1], M2)
    

    rotated_image = cv2.line(rotated_image, (0, y0_low), (rotated_image.shape[1], y0_low), color=(0, 0, 255), thickness=1)
    # rotated_image = cv2.line(rotated_image, (0, y0_high), (rotated_image.shape[1], y0_high), color=(0, 0, 255), thickness=1)
    print(y0_low)
    print(y0_high)
    reduced_image = rotated_image[y0_low:y0_high,:,:]
    # reduced_image = cv2.line(reduced_image, (0, int(window_size/2)), (reduced_image.shape[1], int(window_size/2)), color=(0, 0, 255), thickness=1)
    print(reduced_image.shape)

    rotated_image = cv2.circle(rotated_image, new_point, 1, (0,0,255), 2)
    # rotated_image = cv2.rectangle(rotated_image, (new_point[0]-int(window_size/2),new_point[1]-int(window_size/2)),(new_point[0]+int(window_size/2),new_point[1]+int(window_size/2)), color=(255, 0, 0), thickness=1)
    cv2.imshow('Rotated image', rotated_image)
    cv2.imshow('Original image', img2)
    cv2.imshow('Cropped to the window height', reduced_image)
    cv2.imshow('Section of the cropped image', reduced_image[:,200:220,:])
    cv2.waitKey(0)






def main():

    # # --------------------------------------
    # # Input arguments
    # # --------------------------------------

    # parser = argparse.ArgumentParser(
    #                     prog='From 2D to 3D',
    #                     description='This program gets two 2D images with points and the intrinsic parameters of the camera and returns a 3D plot',
    #                     epilog='Text at the bottom of help')

    # parser.add_argument('-v', '--verbose',
    #                     action='store_true')
    # parser.add_argument('-m', '--matrix',
    #                     type=str, 
    #                     help="YAML file containing the camera matrix (intrinsic parameters)",
    #                     required=True)
    # parser.add_argument('-i1', '--image1',
    #                     type=str, 
    #                     help="Path for the first image", 
    #                     required=True)
    # parser.add_argument('-i2', '--image2',
    #                     type=str, 
    #                     help="Path for the second image", 
    #                     required=True)
    # parser.add_argument('-fm', '--feature_match',
    #                     type=str, 
    #                     help="Path for the second image",
    #                     default="sift", 
    #                     required=False)
    # parser.add_argument('-dist', '--distortion',
    #                     action='store_true',
    #                     help= "Use if you want to use distortion parameters")

    # args = parser.parse_args()

    print('Loading Data...')
    img1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
    img2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
    # img1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Images/Astra/Second_stage/img1_0.png")
    # img2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Images/Astra/Second_stage/img1_1.png")
    K = np.array([[629.400223, 0.000000, 325.240410],
                [0.000000, 627.585852, 262.311140],
                [0.000000, 0.000000, 1.000000]])
    distCoeffs1 = None
    distCoeffs2 = None
    print('Feature matching (calibration)...')
    points1, points2 = feature_match(img1, img2, visualization=True)
    print('Calibration...')
    R, t, F, P1, P2 = calibrate_cameras(points1, points2, K, distCoeffs1, K, distCoeffs2)
    print('Computing and organizing data...')
    epipolar_array, coordinate_array, _ = epipolar_lines(img1, F)
    tempi(epipolar_array[30000], img2, 10, up = 0)
    cv2.destroyAllWindows()
    exit(0)


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