# #!/usr/bin/python3

# # import cv2
# # import numpy as np

# # # Load the left and right images in grayscale
# # left_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", 0)
# # right_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", 0)

# # # Initialize the stereo block matching object 
# # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# # # Compute the disparity map
# # disparity = stereo.compute(left_image, right_image)

# # # Normalize the disparity map for display
# # disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # # Display the disparity map
# # cv2.imshow('Disparity Map', disparity_normalized)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # # Load the left and right images
# # left_image = cv2.imread('left.jpg')
# # right_image = cv2.imread('right.jpg')
# # Load the left and right images in grayscale
# left_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
# right_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
# # Convert to grayscale
# left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
# right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# # Initialize the stereo block matching object
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# # Compute the disparity map
# disparity = stereo.compute(left_gray, right_gray)

# disparity2 = stereo.compute(right_gray, left_gray)


# # Normalize the disparity map
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# disparity2_normalized = cv2.normalize(disparity2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# # Create a color map for the disparity
# disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
# disparity2_color = cv2.applyColorMap(disparity2_normalized, cv2.COLORMAP_JET)
# # Combine the left and right images horizontally
# combined_image = np.hstack((left_image, right_image))

# # Overlay the disparity map on the combined image with transparency
# alpha = 0.5
# # overlay = cv2.addWeighted(combined_image, alpha, np.hstack((disparity_color, disparity_color)), 1 - alpha, 0)
# overlay2 = cv2.addWeighted(combined_image, alpha, np.hstack((disparity2_color, disparity2_color)), 1 - alpha, 0)
# # Display the result
# # cv2.imshow('Disparity Overlay', overlay)
# cv2.imshow('Disparity Overlay 2', overlay2)
# cv2.waitKey(0)import cv2
# import numpy as np

# # # Load the left and right images
# # left_image = cv2.imread('left.jpg')
# # right_image = cv2.imread('right.jpg')
# # Load the left and right images in grayscale
# left_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
# right_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
# # Convert to grayscale
# left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
# right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# # Initialize the stereo block matching object
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# # Compute the disparity map
# disparity = stereo.compute(left_gray, right_gray)

# disparity2 = stereo.compute(right_gray, left_gray)


# # Normalize the disparity map
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# disparity2_normalized = cv2.normalize(disparity2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# # Create a color map for the disparity
# disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
# disparity2_color = cv2.applyColorMap(disparity2_normalized, cv2.COLORMAP_JET)
# # Combine the left and right images horizontally
# combined_image = np.hstack((left_image, right_image))

# # Overlay the disparity map on the combined image with transparency
# alpha = 0.5
# # overlay = cv2.addWeighted(combined_image, alpha, np.hstack((disparity_color, disparity_color)), 1 - alpha, 0)
# overlay2 = cv2.addWeighted(combined_image, alpha, np.hstack((disparity2_color, disparity2_color)), 1 - alpha, 0)
# # Display the result
# # cv2.imshow('Disparity Overlay', overlay)
# cv2.imshow('Disparity Overlay 2', overlay2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2

# # Load the left and right images in grayscale
# left_image_color = cv2.imread("/root/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png")
# right_image_color = cv2.imread("/root/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png")
# left_image = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2GRAY)
# right_image = cv2.cvtColor(right_image_color, cv2.COLOR_BGR2GRAY)

# # Initialize the stereo block matching object
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# # Compute the disparity map
# disparity = stereo.compute(left_image, right_image)

# # Normalize the disparity map for display
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# # Function to find the corresponding pixel in the right image for a pixel in the left image
# def find_corresponding_pixel(x, y, disparity_value):
#     # Calculate the corresponding pixel's x-coordinate in the right image
#     corresponding_x = x - disparity_value
#     return corresponding_x, y

# # Draw lines between corresponding pixels
# def draw_correspondence_lines(left_img_color, right_img_color, disparity_map):
#     # Create a copy of the images to draw lines
#     left_img = cv2.cvtColor(left_img_color, cv2.COLOR_BGR2GRAY)
#     right_img = cv2.cvtColor(right_img_color, cv2.COLOR_BGR2GRAY)
    
#     # Define the number of lines to draw
#     num_lines = 15
    
#     # Calculate the interval at which to draw lines
#     interval = left_img.shape[0] // num_lines
    
#     # Iterate over the selected rows and draw lines
#     for i in range(0, left_img.shape[0], interval):
#         for j in range(left_img.shape[1]):
#             # Get the disparity value for the current pixel
#             disparity_value = disparity_map[i, j]
            
#             # Find the corresponding pixel in the right image
#             corresponding_x, _ = find_corresponding_pixel(j, i, disparity_value)
            
#             # Draw a line between the corresponding pixels
#             if 0 <= corresponding_x < right_img.shape[1]:
#                 cv2.line(left_img_color, (j, i), (corresponding_x, i), (0, 255, 0), 1)
#                 cv2.line(right_img_color, (corresponding_x, i), (j, i), (0, 0, 255), 1)
    
#     # Show the images with lines
#     cv2.imshow('Left Image with Correspondence Lines', left_img_color)
#     cv2.imshow('Right Image with Correspondence Lines', right_img_color)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Call the function to draw lines
# draw_correspondence_lines(left_image_color, right_image_color, disparity_normalized)


import cv2
import numpy as np

# Load the left and right images in grayscale
left_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", 0)
right_image = cv2.imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", 0)

# Compute the stereo block matching
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left_image, right_image)

# Normalize the disparity map for visualization
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Assuming you have a function to find corresponding points
# This is a placeholder for the actual point matching logic
def find_corresponding_points(image1, image2):
    # Placeholder: return random points for demonstration purposes
    points_image1 = np.random.randint(0, min(image1.shape), size=(10, 2))
    points_image2 = np.random.randint(0, min(image2.shape), size=(10, 2))
    return points_image1, points_image2

# Find corresponding points between the two images
points_left, points_right = find_corresponding_points(left_image, right_image)

# Concatenate images horizontally
concatenated_images = np.hstack((left_image, right_image))

# Draw lines between corresponding points
for (point_left, point_right) in zip(points_left, points_right):
    point_right_adjusted = (point_right[0] + right_image.shape[1], point_right[1])
    cv2.line(concatenated_images, tuple(point_left), tuple(point_right_adjusted), (255, 0, 0), 1)

# Display the concatenated image with lines
cv2.imshow('Disparity', concatenated_images)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ... (previous code)

# Convert the left and right images to color so we can display them with colored lines
left_image_color = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
right_image_color = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

# Concatenate images horizontally
concatenated_images = np.hstack((left_image_color, right_image_color))

# Draw lines between corresponding points
for (point_left, point_right) in zip(points_left, points_right):
    point_right_adjusted = (point_right[0] + right_image_color.shape[1], point_right[1])
    cv2.line(concatenated_images, tuple(point_left), tuple(point_right_adjusted), (0, 0, 0), 3)

# Overlay the disparity map on top of the concatenated images
# Convert disparity map to color
disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
# Resize disparity to match the concatenated image size
disparity_resized = cv2.resize(disparity_color, (concatenated_images.shape[1], concatenated_images.shape[0]))
# Blend the disparity map with the concatenated images
alpha = 0.5 # Transparency for the disparity map
blended = cv2.addWeighted(concatenated_images, 1 - alpha, disparity_resized, alpha, 0)

# Display the blended image with lines and disparity map
cv2.imshow('Disparity Overlay', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
