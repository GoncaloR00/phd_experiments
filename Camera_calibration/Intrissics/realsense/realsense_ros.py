#!/usr/bin/env python3

# PhD
# Joao Nuno Valente, DEM, UA
import pyrealsense2 as rs
import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


# Instantiating ROS stuff
topic = '/usb_cam/image_raw'
image_pub = rospy.Publisher(topic,Image, queue_size=10)
bridge = CvBridge()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
rospy.init_node('sender', anonymous=False)

try:
    while not rospy.is_shutdown():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        image_message = bridge.cv2_to_imgmsg(color_image, encoding="mono8")
        image_message.header.stamp = rospy.Time.now()
        image_pub.publish(image_message)

finally:
    # Stop streaming
    pipeline.stop()



# #!/usr/bin/python3

# from pypylon import pylon
# import rospy
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# import cv2


# # Instantiating ROS stuff
# topic = '/usb_cam/image_raw'
# image_pub = rospy.Publisher(topic,Image, queue_size=10)
# bridge = CvBridge()


# # Connecting to the first available camera
# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# # Grabbing continuously (video) with minimal delay
# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
# converter = pylon.ImageFormatConverter()

# # converting to opencv bgr format
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# rospy.init_node('sender', anonymous=False)

# while camera.IsGrabbing() and not rospy.is_shutdown():
#     try:
#         grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#         if grabResult.GrabSucceeded():
#             # Access the image data
#             image = converter.Convert(grabResult)
#             img = image.GetArray()
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             image_message = bridge.cv2_to_imgmsg(img, encoding="mono8")
#             image_message.header.stamp = rospy.Time.now()
#             image_pub.publish(image_message)
#         grabResult.Release()
#     except KeyboardInterrupt:
#        break