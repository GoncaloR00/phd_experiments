import cv2

# Initialize camera objects
camera1 = cv2.VideoCapture(4)  # First camera (usually index 0)
camera2 = cv2.VideoCapture(0)  # Second camera (usually index 1)

while True:
    # Read frames from both cameras
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        print("Error reading frames from cameras.")
        break

    # Display frames (you can modify this part as needed)
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources
camera1.release()
camera2.release()
cv2.destroyAllWindows()
