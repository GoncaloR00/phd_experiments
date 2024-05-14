#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load your images
img1 = cv2.imread('/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png',0)  #queryimage # left image
img2 = cv2.imread('/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png',0) #trainimage # right image

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# Ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# Now we set up our stereo camera with our known parameters
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1,img2)

# Normalize the values to a range from 0..255 for a grayscale image
disparity = cv2.normalize(disparity, disparity, alpha=255,
                          beta=0, norm_type=cv2.NORM_MINMAX)

x, y = np.meshgrid(range(disparity.shape[1]), range(disparity.shape[0]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, disparity)
# Invert axes for better visualization
ax.invert_xaxis()
ax.invert_yaxis()

# Show the plot
plt.show()

# disparity = np.uint8(disparity)

# cv2.imshow('Disparity', disparity)
# cv2.waitKey(0)
# cv2.destroyAllWindows()