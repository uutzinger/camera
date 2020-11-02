##########################################################################
# Testing of blackfly capture thread.
# No display of images
# A data cube has 14 images of 720x540 and is 5.5MBytes in size.
# The camera shuld achieve 525 frames per second.
# The combining 14 frames into a datacube should occur 37.5 times per second
# Captured frames are either transferred through Queue or Shared memory 
##########################################################################
# Results
# =======
# Without Queue:
#   looptime 0.0001
#     5 cubes per second
#     524.2 frames per second
#     CPU Usage: 4-6%
#   looptime 0.001
#     5.0 cubes per second
#     524.2 frames per second
#     CPU Usage: 4-6%
#   looptime 0.0
#     38.8 cubes per second
#     520-524 Frames per second
#     CPU usage: 9-13%
# With Queue:
#   37.4 cubes per second
#   524.2 frames per second
#   CPU Usage: 6-7%
#
##########################################################################
import cv2
import logging
import time
from queue import Queue
import numpy as np

use_queue = True
looptime  = 0.001

# Camera configuration file
from camera.configs.blackfly_configs  import configs

display_interval = 1.0/configs['serverfps']
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,480)
textLocation1    = (10,520)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("backflymain")

# Setting up input and/or output Queue
captureQueue = Queue(maxsize=2)

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
if use_queue:
    camera.start(captureQueue)
else:
    camera.start()

data_cube = np.zeros((14,540,720), dtype=np.uint8)

# Initialize Variables
last_display  = time.time()
last_cps_time = time.time()
measured_cps  = 0
frame_idx     = 0 
num_cubes     = 0

while True:
    current_time = time.time()

    # wait for new image
    if use_queue:
        (frame_time, frame) = captureQueue.get(block=True, timeout=None)
        data_cube[frame_idx,:,:] = frame
        frame_idx += 1
    else:
        if camera.new_frame:
            frame = camera.frame
            data_cube[frame_idx,:,:] = frame
            frame_idx += 1

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes += 1

    if current_time - last_cps_time >= 5.0:
        measured_cps = num_cubes/5.0
        logger.log(logging.DEBUG, "Status:Cubes received per second:{}".format(measured_cps))
        last_cps_time = current_time
        num_cubes = 0

    if not use_queue:
        # make sure the while-loop takes at least looptime to complete
        # since queue is blocking, we dont need this when we use queue
        delay_time = looptime - (time.time() - current_time) 
        if  delay_time >= 0.0001:
            time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

camera.stop()
