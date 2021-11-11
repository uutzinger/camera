##########################################################################
# Testing of rtp stream display 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging
from queue import Queue

window_name = 'RTP'

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

# Setting up input and/or output Queue
captureQueue = Queue(maxsize=32)

# reate camera interface
from camera.capture.rtpcapture import rtpCapture
print("Starting Capture")
camera = rtpCapture(port = 554)
print("Getting Images")
camera.start(captureQueue)

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

while(cv2.getWindowProperty(window_name, 0) >= 0):
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()
