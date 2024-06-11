#!/usr/bin/python3

# import cv2
# import numpy as np

# # Load the left and right images in grayscale
# left_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", 0)
# right_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", 0)

# # Initialize the stereo block matching object 
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# # Compute the disparity map
# disparity = stereo.compute(left_image, right_image)

# # Normalize the disparity map for display
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # Display the disparity map
# cv2.imshow('Disparity Map', disparity_normalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# # Load the left and right images
# left_image = cv2.imread('left.jpg')
# right_image = cv2.imread('right.jpg')
# Load the left and right images in grayscale
left_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
right_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
# Convert to grayscale
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# Initialize the stereo block matching object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(left_gray, right_gray)

# Normalize the disparity map
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Create a color map for the disparity
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

# Combine the left and right images horizontally
combined_image = np.hstack((left_image, right_image))

# Overlay the disparity map on the combined image with transparency
alpha = 0.5
overlay = cv2.addWeighted(combined_image, alpha, np.hstack((disparity_color, disparity_color)), 1 - alpha, 0)

# Display the result
cv2.imshow('Disparity Overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

