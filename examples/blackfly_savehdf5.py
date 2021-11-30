##########################################################################
# Testing of capture and storage thread combined.
# A data cube is 5.5MBytes in size
# It is either copied to shared memory or send via aueue to thread.
# No display of images
##########################################################################
# Results
# =======
# Without Queue:
#   Loop delay 0ms
#     Frame rate: 23
#     Cubes received: 1.8
#     Cubes sent: 1.8
#     Cubes stored: 1.8
#     CPU Usage: 5-6%
#     Disk IO: --
#   Loop delay 1ms 
#     Frame rate: 524.2
#     Cubes received: 4.8
#     Cubes sent: 4.8
#     Cubes stored: 4.8
#     CPU Usage: 4-55
#     Disk IO: 25MB/s
# With Queue: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#   Frame rate: 524.2
#   Cubes received: 37.6
#   Cubes sent: 37.6
#   Cubes stored: 37.4
#   CPU Usage: 5-6%
#   Disk IO: 195-200MB/s
##########################################################################
import logging
import time
import numpy as np
from datetime import datetime

looptime = 0.001
data_cube = np.zeros((14,540,720), dtype=np.uint8)

# Camera configuration file
from configs.blackfly_configs  import configs
    
# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up Storage
from camera.streamer.h5storageserver import h5Server
logger.log(logging.INFO, "Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("C:\\temp\\" + filename)
hdf5.start()

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
logger.log(logging.INFO, "Starting Capture")
camera = blackflyCapture(configs)
camera.start()

# Initialize Variables
frame_idx = 0     # 14 frames is one dataset
num_cubes_sent = 0
num_cubes_received = 0
last_time = time.time()

# Main Loop
while(True):
    current_time = time.time()

    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    data_cube[frame_idx,:,:] = frame
    frame_idx += 1

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_received += 1

        if not hdf5.queue.full():
            hdf5.queue.put_nowait((frame_time, data_cube)) 
            num_cubes_sent += 1
        else:
            logger.log(logging.WARNING, "Status:Storage Queue is full!")

    if current_time - last_time >= 5.0:
        measured_cps = num_cubes_received/5.0
        logger.log(logging.INFO, "Status:Cubes received per second:{}".format(measured_cps))
        measured_cps = num_cubes_sent/5.0
        logger.log(logging.INFO, "Status:Cubes sent per second:{}".format(measured_cps))
        last_time = current_time
        num_cubes_received = 0
        num_cubes_sent = 0

# Cleanup
camera.stop()
hdf5.stop()
