##########################################################################
# Testing of storage server thread.
# No camera involved
##########################################################################
# Results
#   28-33 cubes per second
#   CPU Usage: 1%
##########################################################################
import h5py
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue

data_cube = np.random.randint(0, 255, (14, 540, 720), 'uint8')

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up input and output Queue
storageQueue = Queue(maxsize=5)

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
from camera.streamer.h5storageserver import h5Server
print("Settingup Storage Server")
hdf5 = h5Server("D:\\temp\\" + filename)
print("Starting Storage Server")
hdf5.start(storageQueue)

num_cubes = 0 
last_cps_time = time.time()
cube_time = 0

# Main Loop
while True:
    current_time = time.time()
    storageQueue.put((cube_time, data_cube), block=True, timeout=None)  # Dell Inspiron 7556 achieves 42 to 50 cubes per second
    num_cubes += 1
    cube_time += 1

    if current_time - last_cps_time >= 5.0:
        measured_cps = num_cubes/5.0
        logger.log(logging.DEBUG, "Status:Cubes sent to storeage per second:{}".format(measured_cps))
        last_cps_time = current_time
        num_cubes = 0

# Cleanup
hdf5.stop()
