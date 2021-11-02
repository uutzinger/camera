##########################################################################
# Testing of storage server thread.
# A data cube is 5.5MBytes in size
# It is either copied to shared memory or send via aueue to thread.
# No camera involved
##########################################################################
# Results
# =======
#   48-49 cubes per second with libtiff
#   30 cubes per second with libtiff 
#   Native tiff library is about 13 cubes per second on SSD
##########################################################################
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue

width  = 720      # 1920, 720
height = 540      # 1080, 540
depth  = 14

# synthetic image
data_cube = np.random.randint(0, 255, (depth, height, width), 'uint8')

# Setting up logging
logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up input and output Queue
storageQueue = Queue(maxsize=5)

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".tiff"
from camera.streamer.tiffstorageserver import tiffServer
print("Settingup Storage Server")
tiff = tiffServer("C:\\temp\\" + filename)
print("Starting Storage Server")
tiff.start(storageQueue)

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
tiff.stop()
