##########################################################################
# Testing of storage server thread.
# No camera involved
##########################################################################
# Results
#   450=500 frames per second
#   CPU Usage: 8-10%
##########################################################################
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue

width  = 720      # 1920, 720
height = 540      # 1080, 540
depth  = 3        # only have 3 color planes ?
fps  = 100        # hopeful
size = (width, height)

# synthetic image, needs to be (height, width, depth)
data_cube = np.random.randint(0, 255, (height, width, depth), 'uint8') 

# Setting up logging
logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up input and output Queue
storageQueue = Queue(maxsize=5)

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".mkv"
from camera.streamer.mkvstorageserver import mkvServer
print("Settingup Storage Server")
mkv = mkvServer("C:\\temp\\" + filename, fps, size)
print("Starting Storage Server")
mkv.start(storageQueue)

num_cubes = 0 
cube_time = 0
last_cps_time = time.time()

# Main Loop
while True:
    current_time = time.time()

    storageQueue.put((cube_time, data_cube), block=True, timeout=None) 
    num_cubes += 1

    if current_time - last_cps_time >= 5.0:
        measured_cps = num_cubes/5.0
        logger.log(logging.INFO, "Status:Cubes sent to storeage per second:{}".format(measured_cps))
        last_cps_time = current_time
        num_cubes = 0

# Cleanup
mkv.stop()
