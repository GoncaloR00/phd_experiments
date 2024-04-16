#!/usr/bin/env python3
from pdf2image import convert_from_path
import cv2
import numpy as np

# convert PDF to image then to array ready for opencv
pages = convert_from_path('random_points.pdf')
img = np.array(pages[0])

# opencv code to view image
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imwrite("random_points.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
