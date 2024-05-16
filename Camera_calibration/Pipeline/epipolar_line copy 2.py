#!/usr/bin/python3

import numpy as np
import cv2
from random import randint
import copy
import json
import open3d as o3d
import gc

# np.set_printoptions(threshold=np.inf)
x_mouse = None
y_mouse = None
old_x_mouse = None
old_y_mouse = None


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
    print(f"Rotation: \n{R} \nTranslation: \n{t}")
    # Create projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    # Apply the camera intrinsics
    P1 = cameraMatrix1 @ P1
    P2 = cameraMatrix2 @ P2
    # return P1, P2
    return R, t, F, P1, P2

def triangulate_and_plot(P1, P2, points1, points2):
    # Triangulate points
    points_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3D = points_hom / points_hom[3]
    points_3D_np = np.array(points_3D[:3, :].T, dtype=np.float64)
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D_np)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
    global x_mouse, y_mouse
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        x_mouse = x
        y_mouse = y
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
        x_mouse = x
        y_mouse = y

def find_matching_point(img1, img2, point, epiline, window_size=5, l_ratio = 0.8, similarity_func=cv2.norm):
    # Initialize best match (lowest distance)
    best_match = None
    second_dist = None
    best_distance = float('inf')

    a, b, c = epiline

    # Iterate over each point in the epipolar line
    for x in range(0, img2.shape[1]):
        if b != 0:
            y = -1*(a*x + c) / b
            y = int(round(y))  # Ensure y is an integer for indexing
        else:
            continue  # If line is vertical, skip this iteration

        # Check if y is within image bounds
        if y < 0 or y >= img2.shape[0]:
            continue
        
        a1 = point[1]-window_size
        b1 = point[1]+window_size
        c1 = point[0]-window_size
        d1 = point[0]+window_size
        a2 = y-window_size
        b2 = y+window_size
        c2 = x-window_size
        d2 = x+window_size

        if a1 < 0 or c1 <0 or a2 <0 or c2 <0 or b1 > img1.shape[0] or d1 > img1.shape[1] or b2 > img2.shape[0] or d2 > img2.shape[1]:
            continue

        # Compute similarity measure
        distance = similarity_func(img1[a1:b1, c1:d1],
                                img2[a2:b2, c2:d2], cv2.NORM_L1)

        # Update best match if better
        if distance < best_distance:
            second_dist = best_distance
            best_distance = distance
            best_match = (x, y)
        #     print(f"img1[{point[1]-window_size}:{point[1]+window_size}, {point[0]-window_size}:{point[0]+window_size}], img2[{y-window_size}:{y+window_size}, {x-window_size}:{x+window_size}")
    print(f"Best distance = {best_distance}\nSecond = {second_dist}")
    if second_dist == float('inf') or second_dist is None:
        best_match = None
    # elif best_distance > 0.8*second_dist:
    elif not(best_distance < l_ratio*second_dist):
        print('invalid')
        best_match = None

    return best_match

if __name__=="__main__":
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

    with open("points1.json", 'r') as f:
        points1 = np.array(json.load(f))
    with open("points2.json", 'r') as f:
        points2 = np.array(json.load(f))
    # points1 = points1[5000:8000]
    # points2 = points2[5000:8000]
    
    # check_points1 = points1[points1==np.nan]
    # points1 = points1[3000:3060]
    # points2 = points2[3000:3060]
    # print(points1)
    triangulate_and_plot(P1, P2, points1, points2)
    # close the window 
    # cv2.destroyAllWindows()