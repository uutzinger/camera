import cv2
import logging
import time
import platform
from queue import Queue
import numpy as np

use_queue = True
looptime         = 0.0

# Dell Inspiron 15 internal camera
# from camera.configs.dell_internal_configs  import configs as configs0

# Eluktronics Max-15 internal camera
from camera.configs.eluk_configs  import configs as configs0
#
# Nano Jetson IMX219 camera
# from camera.configs.nano_IMX219_configs  import configs as configs0
#
# Raspberry Pi v1 & v2 camera
# from camera.configs.raspi_v1module_configs  import configs as configs0
# from camera.configs.raspi_v2module_configs  import configs as configs0
#
# ELP
# from camera.configs.ELP1080p_configs  import configs as configs0
#
 
if configs0['displayfps'] >= configs0['fps'] and use_queue:
    display_interval = 0
else:
    display_interval = 1.0/configs0['displayfps']

window_name      = 'Camera0'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,60)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL


# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("CV2Capture")

# Setting up input and/or output Queue
captureQueue0 = Queue(maxsize=32)

# Create camera interface based on computer OS you are running
plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs0)
elif plat == 'Linux':
    if platform.machine() == "aarch64":
        from camera.capture.nanocapture import nanoCapture
        camera0 = nanoCapture(configs0)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l':
        from camera.capture.cv2capture import cv2Capture
        camera0 = cv2Capture(configs0)
        # from picapture import piCapture
        # camera = piCapture()
elif plat == 'MacOS':
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs0)
else:
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs0)

#print("CV2 Capture Options")
#camera.cv2SettingsDebug()

print("Getting Images")
if use_queue:
    camera0.start(captureQueue0)
else:
    camera0.start()

# Initialize Variables
last_display  = time.time()
last_fps_time  = time.time()
measured_dps  = 0
num_frames_received    = 0
num_frames_displayed   = 0

while(cv2.getWindowProperty(window_name, 0) >= 0):
    current_time = time.time()
    # wait for new image
    if use_queue:
        (frame_time, frame) = captureQueue0.get(block=True, timeout=None)
        num_frames_received += 1
    else:
        if camera0.new_frame:
            frame = camera0.frame
            frame_time = camera0.frame_time
            num_frames_received +=1

    if current_time - last_fps_time >= 5.0:
        measured_fps = num_frames_received/5.0
        logger.log(logging.DEBUG, "Status:Frames received per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

    if (current_time - last_display) >= display_interval:
        cv2.putText(frame,"Capture FPS:{} [Hz]".format(camera0.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_display = current_time
        num_frames_displayed += 1

    if not use_queue:
        # make sure the while-loop takes at least looptime to complete
        # since queue is blocking, we dont need this when we use queue
        delay_time = looptime - (time.time() - current_time) 
        if  delay_time >= 0.001:
            time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

camera0.stop()
cv2.destroyAllWindows()
