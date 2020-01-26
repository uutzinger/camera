import cv2
import logging
import time
import platform

from cv2capture import cv2Capture

logging.basicConfig(level=logging.DEBUG)

# reate camera interface
plat = platform.system()
if plat == 'Windows': 
    from cv2capture import cv2Capture
    camera = cv2Capture()
elif plat == 'Linux':
    if platform.machine() == "aarch64":
        from nanocapture import nanoCapture
        camera = nanoCapture()
    elif platform.machine() == "armv6l":
        from picapture import piCapture
        camera = piCapture()
elif plat == 'MacOS':
    from cv2capture import cv2Capture
    camera = cv2Capture()
else:
    from cv2Capture import cv2Capture
    camera = cv2Capture()

print("Starting Capture")
camera.start()

print("Getting Frames")

window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
last_fps_time = time.time()
num_frames = 0
while(cv2.getWindowProperty("Camera", 0) >= 0):
    if camera.new_frame:
        cv2.imshow('Camera', camera.frame)
        num_frames += 1
    current_time = time.time()
    if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
        print("CaptureFPS: ", num_frames/5.0)
        num_frames = 0
        last_fps_time = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.stop()
cv2.destroyAllWindows()
