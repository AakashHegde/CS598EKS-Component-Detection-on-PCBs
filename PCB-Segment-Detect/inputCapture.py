from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
from libcamera import controls
import time
import cv2

from segmentAndDetect import split_and_display, detect_components

picam2 = Picamera2()  ## Create a camera object

# set camera video resolution
dispW = 640
dispH = 640
picam2.preview_configuration.main.size = (dispW,dispH)

## since OpenCV requires RGB configuration we set the same format for picam2. The 888 implies # of bits on Red, Green and Blue
picam2.preview_configuration.main.format= "RGB888"
picam2.preview_configuration.align() ## aligns the size to the closest standard format
picam2.preview_configuration.controls.FrameRate=30 ## set the number of frames per second
picam2.preview_configuration.queue = False

picam2.configure("preview")
picam2.start()
# allow the camera to warmup
time.sleep(0.1)

# auto-focus
picam2.set_controls({"AfMode": 2, "AfTrigger": 0})

while True:	
	frame = picam2.capture_array()

	# ~ cv2.imshow("Camera Frame", frame)
	cv2.imshow("Camera Frame", detect_components(frame))
	# ~ time.sleep(0.5)

	key=cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
picam2.stop()
