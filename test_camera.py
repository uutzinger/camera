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
        camera = cv2Capture()
        # from picapture import piCapture
        # camera = piCapture()
elif plat == 'MacOS':
    from cv2capture import cv2Capture
    camera = cv2Capture()
else:
    from cv2capture import cv2Capture
    camera = cv2Capture()

print("CV2 Capture Options")
camera.cv2SettingsDebug()

print("Starting Capture")
camera.start()

print("Getting Frames")

doshow = False

window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE) # or normal
last_fps_time = time.time()
while(cv2.getWindowProperty("CSI Camera", 0) >= 0):
    if doshow:
        if camera.new_frame:
            cv2.imshow('CSI Camera', camera.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        pass
camera.stop()
cv2.destroyAllWindows()
