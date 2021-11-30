##########################################################################
# Testing of rtsp stream display 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging

from configs.rtsp_configs import configs as configs

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("RTSP")

# reate camera interface
from camera.capture.rtspcapture import rtspCapture
# camera = rtspCapture(rtsp='rtsp://192.168.8.50:8554/unicast')
# camera = rtspCapture(rtsp='rtsp://10.41.83.100:554/camera')
logger.log(logging.INFO, "Starting Capture")
camera = rtspCapture(configs, rtsp='rtsp://127.0.0.1:554')
logger.log(logging.INFO, "Getting Images")
camera.start()

window_handle = cv2.namedWindow("RTSP", cv2.WINDOW_NORMAL)
while(cv2.getWindowProperty("RTSP", 0) >= 0):
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    cv2.imshow('RTSP', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.stop()
cv2.destroyAllWindows()
