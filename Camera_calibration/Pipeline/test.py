#!/usr/bin/python3

#!/usr/bin/env python3
import numpy as np
import cv2
from random import randint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy


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
    return R, t

def rectify_images(img1, img2):
    K = np.array([[629.400223, 0.000000, 325.240410],
                [0.000000, 627.585852, 262.311140],
                [0.000000, 0.000000, 1.000000]])
    distCoeffs1 = None
    distCoeffs2 = None
    points1, points2 = feature_match(img1, img2, visualization=True)
    R, T = calibrate_cameras(points1, points2, K, distCoeffs1, K, distCoeffs2)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, distCoeffs1, K, distCoeffs2, img1.shape[::-1], R, T)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(K, distCoeffs1, R1, P1, img1.shape[::-1], cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(K, distCoeffs2, R2, P2, img2.shape[::-1], cv2.CV_32FC1)
    img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)
    return img_rect1, img_rect2, Q


# Load your images
imgL = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", cv2.IMREAD_GRAYSCALE)

# Rectify your images (you'll need to use your camera parameters here)
imgL, imgR, Q = rectify_images(imgL, imgR)

# Perform stereo matching (using StereoSGBM in this example)
window_size = 3
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 16,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# disparity = np.uint8(disparity)

# cv2.imshow('Disparity', disparity)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# np.set_printoptions(threshold=np.inf)

# disparity = cv2.normalize(disparity, disparity, alpha=255,
#                           beta=0, norm_type=cv2.NORM_MINMAX)
# disparity[disparity==0]=np.nan
# # print(disparity)

# x, y = np.meshgrid(range(disparity.shape[1]), range(disparity.shape[0]))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, disparity)
# # Invert axes for better visualization
# ax.invert_xaxis()
# ax.invert_yaxis()

# # Show the plot
# plt.show()


# Assuming you have a disparity map
# disparity = ...

# Load your disparity map
# disparity = cv2.imread('disparity.jpg', 0)

# Assuming you have the Q matrix from stereo rectification
# Q = ...

# Reproject the disparity image to 3D
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Get the colors from the original image
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

# Mask out background points
mask_map = disparity > disparity.min()

# Apply the mask to the points and colors
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(output_points[:, 0], output_points[:, 1], output_points[:, 2], c=output_colors/255, s=3)
plt.show()
