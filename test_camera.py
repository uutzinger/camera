import cv2
import logging
import time
import platform

# Show images in window
doshow = True

# Camera configuration file
from camera.configs.dell_internal_configs  import configs

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING

# Create camera interface
# Based on computer OS you are running
plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)
elif plat == 'Linux':
    if platform.machine() == "aarch64":
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l':
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs)
        # from picapture import piCapture
        # camera = piCapture()
elif plat == 'MacOS':
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)
else:
    from camera.cappture.cv2capture import cv2Capture
    camera = cv2Capture(configs)

# print("CV2 Capture Options")
# camera.cv2SettingsDebug()

print("Starting Capture")
camera.start()

print("Getting Frames")

window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE) # or normal
last_fps_time = time.time()
while(cv2.getWindowProperty("Camera", 0) >= 0):
    if doshow:
        if camera.new_frame:
            cv2.imshow('Camera', camera.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        pass
camera.stop()
cv2.destroyAllWindows()
