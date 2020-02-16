import cv2
import logging
import time
import platform

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
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l':
        from cv2capture import cv2Capture
        camera = cv2Capture(1)
        # from picapture import piCapture
        # camera = piCapture()
elif plat == 'MacOS':
    from cv2capture import cv2Capture
    camera = cv2Capture()
else:
    from cv2capture import cv2Capture
    camera = cv2Capture()

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
