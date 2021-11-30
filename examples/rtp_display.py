##########################################################################
# Testing of rtp stream display 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging

window_name = 'RTP'

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

# reate camera interface
from camera.capture.rtpcapture import rtpCapture
print("Starting Capture")
camera = rtpCapture(port = 554)
print("Getting Images")
camera.start()

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

while(cv2.getWindowProperty(window_name, 0) >= 0):
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.stop()
cv2.destroyAllWindows()
