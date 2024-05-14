#!/usr/bin/python3

from pypylon import pylon
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2


# Instantiating ROS stuff
topic = '/usb_cam/image_raw'
image_pub = rospy.Publisher(topic,Image, queue_size=10)
bridge = CvBridge()


# Connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabbing continuously (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

rospy.init_node('sender', anonymous=False)

while camera.IsGrabbing() and not rospy.is_shutdown():
    try:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_message = bridge.cv2_to_imgmsg(img, encoding="mono8")
            image_message.header.stamp = rospy.Time.now()
            image_pub.publish(image_message)
        grabResult.Release()
    except KeyboardInterrupt:
       break
    
