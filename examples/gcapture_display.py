##########################################################################
# GStreamer (gCapture) display + capture example
#
# Uses camera.capture.gcapture.gCapture which selects a reasonable GStreamer
# source (libcamerasrc/nvarguscamerasrc/v4l2src) based on availability.
#
# You need to set the proper config import below.
##########################################################################

import logging
import time

from queue import Empty

import cv2


# Can import or define camera configuration below
# Make sure you have configs folder in the folder of your main program
# -Eluktronics Max-15 internal camera
from configs.eluk_configs import configs as configs
# -Nano Jetson IMX219 camera
# from configs.nano_IMX219_configs  import configs as configs
# -Raspberry Pi v1 & v2 camera
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs


camera_index = 0

if configs['displayfps'] >= 0.8 * configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0 / configs['displayfps']

dps_measure_time = 5.0  # assess performance every 5 secs

window_name = 'gCapture Camera'
font = cv2.FONT_HERSHEY_SIMPLEX
textLocation0 = (10, 20)
textLocation1 = (10, 60)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("gcapture_display")

from camera.capture.gcapture import gCapture

camera = gCapture(configs, camera_num=camera_index)
if not getattr(camera, 'cam_open', False):
    # still start so it can emit logs, but we likely won't receive frames
    logger.log(logging.WARNING, "gCapture did not open the camera (check logs/pipeline)")

logger.log(logging.INFO, "Getting Images")
camera.start()

last_display = time.perf_counter()
last_fps_time = time.perf_counter()
measured_dps = 0
num_frames_received = 0
num_frames_displayed = 0

stop = False
frame = None
while not stop:
    current_time = time.perf_counter()

    try:
        (_, frame) = camera.capture.get(timeout=0.25)
        num_frames_received += 1
    except Empty:
        pass

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

    if (frame is not None) and ((current_time - last_display) >= display_interval):
        frame_display = frame.copy()
        cv2.putText(frame_display, "Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display, "Display FPS:{} [Hz]".format(measured_dps), textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
            stop = True
        last_display = current_time
        num_frames_displayed += 1

camera.stop()
try:
    camera.join(timeout=2.0)
except Exception:
    pass
try:
    camera.close_cam()
except Exception:
    pass
cv2.destroyAllWindows()
