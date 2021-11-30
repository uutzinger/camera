##########################################################################
# Testing of storage server thread.
# No camera involved
##########################################################################
# Results
#   17-48 cubes per second on SSD drive
#   21-24 cubes per second on NVME drive
#   CPU Usage: 1%
##########################################################################
import logging
import time
import numpy as np
from datetime import datetime

width  = 720      # 1920, 720
height = 540      # 1080, 540
depth  = 14       # 
image = np.random.randint(0, 255, (depth, height, width), 'uint8') 

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
from camera.streamer.h5storageserver import h5Server
logger.log(logging.INFO, "Settingup Storage Server")
hdf5 = h5Server("C:\\temp\\" + filename)
logger.log(logging.INFO, "Starting Storage Server")
hdf5.start()

num_images = 0 
last_time = time.time()

# Main Loop
while True:
    current_time = time.time()
    hdf5.queue.put((current_time, image), block=True, timeout=None)
    num_images += 1

    while not hdf5.log.empty():
        (level, msg)=hdf5.log.get_nowait()
        logger.log(level, "Status:{}".format(msg))

    if current_time - last_time >= 5.0:
        measured_fps = num_images/5.0
        logger.log(logging.INFO, "Status:Cubes sent to storeage per second:{}".format(measured_fps))
        last_time = current_time
        num_images = 0

    time.sleep(0.001)

# Cleanup
hdf5.stop()
