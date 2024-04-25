#!/usr/bin/python3

# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(2) 
first = True
counter = 0
while(True): 
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read()
    if first:
        first = False
        print(f"Shape = {frame.shape}")
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    
    if 0xFF == ord('s'):
        counter += 1
        cv2.imwrite(f'img{counter}', frame)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
