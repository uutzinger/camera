##########################################################################
# OpenCV (cv2Capture) display + capture example
#
# Notes:
# - On Jetson Nano (aarch64) this will use nanoCapture (nvargus) for CSI.
# - On other platforms it uses cv2Capture.
#
# You need to set the proper config import below.
##########################################################################

import logging
import time
import platform
import os
import sys

if sys.platform.startswith('win'):
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from camera.utils import probeCameras

loop_interval = 1.0/200.0

# Camera Signature
# In case we have multiple camera, we can search for default driver settings
# and compare to camera signature, opencv unfortunately does not return the
# serial number of the camera
# Example: Generic Webcam: 640, 480, YUYV
# Example: FLIRLepton: 160, 120, BGR3
widthSig = 640
heightSig = 480
fourccSig = "\x16\x00\x00\x00"

# default camera starts at 0 by operating system
camera_index = 0
# Scan up to 5 cameras
camprops = probeCameras(5)
# Try to find the one that matches our signature
score = 0
for i in range(len(camprops)):
    try:
        found_fourcc = 1 if camprops[i]['fourcc'] == fourccSig else 0
    except Exception:
        found_fourcc = 0
    try:
        found_width = 1 if camprops[i]['width'] == widthSig else 0
    except Exception:
        found_width = 0
    try:
        found_height = 1 if camprops[i]['height'] == heightSig else 0
    except Exception:
        found_height = 0
    tmp = found_fourcc + found_width + found_height
    if tmp > score:
        score = tmp
        camera_index = i

# Can import or define camera configuration below
# Make sure you have configs folder in the folder of your main program
# -Eluktronics Max-15 internal camera
from configs.eluk_configs import configs as configs
# -Dell Inspiron 15 internal camer
# from configs.dell_internal_configs  import configs as configs
# -Generic webcam
# from configs.generic_1080p import configs as configs
# -Nano Jetson IMX219 camera
# from configs.nano_IMX219_configs  import configs as configs
# -Raspberry Pi v1 & v2 camera
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs
# -ELP MAX15 internal camera
# from configs.ELP1080p_configs  import configs as configs
# -FLIR Lepton 3.5
# from configs.FLIRlepton35 import confgis as configs

if configs['displayfps'] >= 0.8 * configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0 / configs['displayfps']

dps_measure_time = 5.0  # assess performance every 5 secs

window_name = 'Camera'
font = cv2.FONT_HERSHEY_SIMPLEX
textLocation0 = (10, 20)
textLocation1 = (10, 60)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG)  # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("cv2_capture_display")

# Create camera interface based on computer OS you are running
plat = platform.system()
if plat == 'Linux' and platform.machine() == "aarch64":  # Jetson Nano
    from camera.capture.nanocapture import nanoCapture

    camera = nanoCapture(configs, camera_index)
else:
    from camera.capture.cv2capture import cv2Capture

    camera = cv2Capture(configs, camera_index)

logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
last_display = time.perf_counter()
last_fps_time = time.perf_counter()
measured_dps = 0
num_frames_received = 0
num_frames_displayed = 0

while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
    current_time = time.perf_counter()

    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    num_frames_received += 1
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, "{}".format(msg))

    if (current_time - last_fps_time) >= dps_measure_time:
        measured_fps = num_frames_received / dps_measure_time
        logger.log(logging.INFO, "MAIN:Frames received per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed / dps_measure_time
        logger.log(logging.INFO, "MAIN:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

    if (current_time - last_display) >= display_interval:
        frame_display = frame.copy()
        cv2.putText(frame_display, "Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display, "Display FPS:{} [Hz]".format(measured_dps), textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame_display)
        # quit the program if users enter q or closes the display window
        # the waitKey function limits the display frame rate
        # without waitKey the opencv window is not refreshed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_display = current_time
        num_frames_displayed += 1

# Clean up
camera.stop()
cv2.destroyAllWindows()
