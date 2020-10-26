import cv2
import logging
import time
import platform

logging.basicConfig(level=logging.DEBUG)

# reate camera interface
from camera.capture.rtspcapture import rtspCapture
# camera = rtspCapture(rtsp='rtsp://192.168.8.50:8554/unicast')
print("Starting Capture")
camera = rtspCapture(rtsp='rtsp://10.41.83.100:554/camera')
print("Getting Images")
camera.start()

window_handle = cv2.namedWindow("RTSP", cv2.WINDOW_NORMAL)
while(cv2.getWindowProperty("RTSP", 0) >= 0):
    if camera.new_frame:
        cv2.imshow('RTSP', camera.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.stop()
cv2.destroyAllWindows()
