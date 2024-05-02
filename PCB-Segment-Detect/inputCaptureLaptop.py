import cv2
import time
from segmentAndDetect import detect_components

# Set camera video resolution
dispW = 640
dispH = 640

print('1')

# Create a video capture object
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

print('2')
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()


print('3')

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS if needed

# Allow the camera to warmup
time.sleep(0.1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Call the detection components function
    frame = detect_components(frame)

    # Display the resulting frame
    cv2.imshow('Camera Frame', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
