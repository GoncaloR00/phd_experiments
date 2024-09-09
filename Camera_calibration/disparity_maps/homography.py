import cv2
import numpy as np

# Load the images
image1 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Create a BFMatcher (Brute-Force Matcher)
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test to get good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.3 * n.distance:
        good_matches.append(m)

# Extract corresponding points
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate homography
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp image1 to align with image2
aligned_image1 = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))

# Find common region
common_region = cv2.bitwise_and(aligned_image1, image2)
unique_to_image2 = cv2.bitwise_not(aligned_image1) & image2

foreground = cv2.absdiff(image2, aligned_image1)

# Invert the foreground to get background (unique to image1)
background = cv2.bitwise_not(foreground)
print(background)
# Show the common region
cv2.imshow('aligned image 1', aligned_image1)
cv2.imshow('Common Region', common_region)
# cv2.imshow('Unique to image 1', background)
# cv2.imshow('Unique to image 2', unique_to_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
