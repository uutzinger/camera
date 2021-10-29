##########################################################################
# Testing of blackfly capture thread.
# No display of images
# A data cube has 14 images of 720x540 and is 5.5MBytes in size.
# The camera shuld achieve 525 frames per second.
# The combining 14 frames into a datacube should occur 37.5 times per second
##########################################################################
# Results
# =======
#   37.4 cubes per second
#   524.2 frames per second
#   CPU Usage: 6-7%
##########################################################################
import logging
import time
from queue import Queue
import numpy as np

loop_interval  = 1./1000.

# Camera configuration file
from configs.blackfly_configs  import configs

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
camera.start(captureQueue)

data_cube = np.zeros((14,540,720), dtype=np.uint8)

# Initialize Variables
last_cps_time = time.time()
measured_cps  = 0
frame_idx     = 0 
num_cubes     = 0

while True:
    current_time = time.time()

    # wait for new image
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)
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
        
camera.stop()
