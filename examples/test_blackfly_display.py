##########################################################################
# Testing of blackfly capture thread.
# The camera shuld achieve 525 frames per second.
# Captured frames are either transferred through Queue or Shared memory 
##########################################################################
# Results
# =======
# Without Queue:
#   looptime 0.0
#    Display Interval 1.0
#      Capture FPS 524.2
#      Frames retrieved 513-518/s
#      Display FPS 1.0
#      CPU Usage: 7-8%
#      --
#     Display Interval 0.03
#      Capture FPS 524.2
#      Frames retrieved 257-283/s
#      Display FPS 32,6
#      CPU Usage: 7-8%
#   looptime 0.001
#     Display Interval 1.0
#      Capture FPS 524.2
#      Frames retrieved 510/s
#      Display FPS 1.0
#      CPU usage: 8 %
#      --
#     Display Interval 0.03
#      Capture FPS 524.4
#      Frames retrieved 280/s
#      Display FPS 32.6
#      CPU usage: 7-8 %
# With Queue:  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#  Display Interval 0.03
#    Capture FPS: 524.2
#    Display FPS: 32.2
#    CPU Usage: 6-7%
#  Display Interval 1.0
#    Capture FPS: 524.2
#    Display FPS: 1.0
#    CPU Usage: 5%
#
##########################################################################
import cv2
import logging
import time
from queue import Queue
import numpy as np

# Camera configuration file
from configs.blackfly_configs  import configs

display_interval = 1.0/configs['displayfps']
window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,20)
textLocation1  = (10,60)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Blackfly")

# Setting up input and/or output Queue
captureQueue = Queue(maxsize=128)

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
camera.start(captureQueue)

# Initialize Variables
last_display  = time.time()
last_fps_time = time.time()
measured_dps  = 0
num_frames_received  = 0
num_frames_displayed = 0

while (cv2.getWindowProperty(window_name, 0) >= 0):
    current_time = time.time()
    # wait for new image
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)
    num_frames_received += 1

    if current_time - last_fps_time >= 5.0:
        measured_fps = num_frames_received/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

    if (current_time - last_display) > display_interval:
        display_frame = frame.copy()
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, display_frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_display = current_time
        num_frames_displayed += 1

camera.stop()
cv2.destroyAllWindows()
