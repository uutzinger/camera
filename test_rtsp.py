import cv2
import logging
import time
import platform

logging.basicConfig(level=logging.DEBUG)

# reate camera interface
from rtspcapture import rtspCapture
camera = rtspCapture(rtsp='rtsp://192.168.8.50:554/')

print("Starting Capture")
camera.start()

print("Getting Frames")

window_handle = cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
last_fps_time = time.time()
while(cv2.getWindowProperty("Camera", 0) >= 0):
    if camera.new_frame:
        cv2.imshow('Camera', camera.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.stop()
cv2.destroyAllWindows()
