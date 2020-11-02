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
import h5py
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue

looptime = 0.001
use_queue = True
data_cube = np.zeros((14,540,720), dtype=np.uint8)

# Camera configuration file
from camera.configs.blackfly_configs  import configs
    
# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up input and output Queue
captureQueue = Queue(maxsize=32)
storageQueue = Queue(maxsize=2)

# Setting up Storage
from camera.streamer.storageserver import h5Server
print("Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("C:\\temp\\" + filename)
print("Starting Storage Server")
if use_queue: 
    hdf5.start(storageQueue)
else:
    hdf5.start()

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
if use_queue:
    camera.start(captureQueue)
else:
    camera.start()

# Initialize Variables
frame_idx = 0     # 14 frames is one dataset
num_cubes_sent = 0
num_cubes_received = 0
last_cps_time = time.time()

# Main Loop
while(True):
    current_time = time.time()

    # wait for new image
    if use_queue:
        (frame_time, frame) = captureQueue.get(block=True, timeout=None)
        data_cube[frame_idx,:,:] = frame
        frame_idx += 1
    else:
        if camera.new_frame:
            frame = camera.frame
            frame_time = camera.frame_time
            data_cube[frame_idx,:,:] = frame
            frame_idx += 1

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_received += 1

        if use_queue:
            if not storageQueue.full():
                storageQueue.put((frame_time, data_cube), block=False) 
                num_cubes_sent += 1
            else:
                self.logger.log(logging.DEBUG, "Status:Storage Queue is full!")
        else:
            hdf5.framearray_time = frame_time # data array will be saved with this as its name
            hdf5.framearray = data_cube # transfer data array to storage server
            num_cubes_sent += 1

    if current_time - last_cps_time >= 5.0:
        measured_cps = num_cubes_received/5.0
        logger.log(logging.DEBUG, "Status:Cubes received per second:{}".format(measured_cps))
        measured_cps = num_cubes_sent/5.0
        logger.log(logging.DEBUG, "Status:Cubes sent per second:{}".format(measured_cps))
        last_cps_time = current_time
        num_cubes_received = 0
        num_cubes_sent = 0

    if not use_queue:
        # make sure the while-loop takes at least looptime to complete
        # since queue is blocking, we dont need this when we use queue
        delay_time = looptime - (time.time() - current_time) 
        if  delay_time >= 0.001:
            time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

# Cleanup
camera.stop()
hdf5.stop()
