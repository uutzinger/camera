##########################################################################
# Testing of storage server thread.
# A data cube is 5.5MBytes in size
# It is either copied to shared memory or send via aueue to thread.
# No camera involved
##########################################################################
# Results
# =======
#   35-45 cubes per second with libtiff onto SSD
#   13    cubes per second without libtiff onton SSD
#   20-23 cubes per second with libtiff onto NVME
##########################################################################
import logging
import time
import numpy as np
from datetime import datetime

width  = 720      # 1920, 720
height = 540      # 1080, 540
depth  = 14

# synthetic image
data_cube = np.random.randint(0, 255, (depth, height, width), 'uint8')

# Setting up logging
logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".tiff"
from camera.streamer.tiffstorageserver import tiffServer
logger.log(logging.INFO, "Settingup Storage Server")
tiff = tiffServer("C:\\temp\\" + filename)
logger.log(logging.INFO, "Starting Storage Server")
tiff.start()

num_cubes = 0 
last_time = time.time()

# Main Loop
while True:
    current_time = time.time()

    tiff.queue.put((current_time, data_cube), block=True, timeout=None) 
    num_cubes += 1

    if (current_time - last_time) >= 5.0:
        measured_cps = num_cubes/5.0
        logger.log(logging.INFO, "Status:Cubes sent to storeage per second:{}".format(measured_cps))
        last_time = current_time
        num_cubes = 0

    while not tiff.log.empty():
        (level, msg)=tiff.log.get_nowait()
        logger.log(level, "Status:{}".format(msg))

# Cleanup
tiff.stop()
