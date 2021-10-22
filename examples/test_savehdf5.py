##########################################################################
# Testing of storage server thread.
# A data cube is 5.5MBytes in size
# It is either copied to shared memory or send via aueue to thread.
# 
##########################################################################
# Results
# =======
# Without Threading
#   Loop delay 0ms
#     48-49 cubes sent from storage server to disk, starts with 180cubes then goes down to within 20secs
#     CPU Usage: 2-3%
#     DISK I/O: 250MB/s, starts with 450MB/s
# Without Queue:
#   Loop delay 0ms
#     280'000 cubes per second send to storage server
#     28-29 sent from storage server to disk
#     CPU Usage: 18-20%
#     DISK I/O: 145MB/s
#   Loop delay 1ms 
#     64.4 cubes per second to storage thread
#     47-48 cubes per second to disk
#     CPU Usage: 1.5-2.5%
#     DISK I/O:  240MB/s
# With Queue:
#   48-49 cubes per second
#   CPU Usage: 1-2.6%
#   Disk IO: 250MB/s
#   Loop delay does not change results
##########################################################################
import h5py
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue

looptime = 0.001
use_queue = True
data_cube = np.random.randint(0, 255, (14, 540, 720), 'uint8')

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up input and output Queue
storageQueue = Queue(maxsize=5)

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
from camera.streamer.storageserver import h5Server
print("Settingup Storage Server")
hdf5 = h5Server("C:\\temp\\" + filename)
print("Starting Storage Server")
if use_queue: 
    hdf5.start(storageQueue)
else:
    hdf5.start()

num_cubes = 0 
last_cps_time = time.time()
cube_time = 0

# Main Loop
while True:
    current_time = time.time()
    if use_queue:
        storageQueue.put((cube_time, data_cube), block=True, timeout=None)  # Dell Inspiron 7556 achieves 42 to 50 cubes per second
    else:
        hdf5.framearray_time = cube_time # data array will be saved with this as its name, Dell Inspiron 7556 achieves about 46 cubes per second
        hdf5.framearray = data_cube # transfer data array to storage server
    num_cubes += 1
    cube_time += 1

    if current_time - last_cps_time >= 5.0:
        measured_cps = num_cubes/5.0
        logger.log(logging.DEBUG, "Status:Cubes sent to storeage per second:{}".format(measured_cps))
        last_cps_time = current_time
        num_cubes = 0

    if not use_queue: # We dont need loop delay when using queue as it is blocking
        delay_time = looptime - (time.time() - current_time) # make sure the while-loop takes at least looptime to complete
        if  delay_time >= 0.001:
            time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

# Cleanup
hdf5.stop()
