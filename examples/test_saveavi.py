##########################################################################
# Testing of storage server thread.
# No camera involved
##########################################################################
# Results
# =======
#   95-100 frames per second
#   CPU Usage: 8-10%
##########################################################################
import logging
import time
import numpy as np
from datetime import datetime

width  = 720      # 1920, 720
height = 540      # 1080, 540
depth  = 3        # only have 3 color planes ?
fps  = 100        # hopeful
size = (width, height)

# synthetic image, needs to be (height, width, depth)
image = np.random.randint(0, 255, (height, width, depth), 'uint8') 

# Setting up logging
logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up Storage
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".avi"
from camera.streamer.avistorageserver import aviServer
logger.log(logging.INFO, "Setting up Storage Server")
avi = aviServer("C:\\temp\\" + filename, fps, size)
logger.log(logging.INFO, "Starting Storage Server")
avi.start()

num_images = 0 
last_time = time.time()

# Main Loop
while True:
    current_time = time.time()

    avi.queue.put((current_time, image), block=True, timeout=None) 
    num_images += 1

    while not avi.log.empty():
        (level, msg)=avi.log.get_nowait()
        logger.log(level, "Status:{}".format(msg))

    if current_time - last_time >= 5.0:
        measured_fps = num_images/5.0
        logger.log(logging.INFO, "Status:Cubes sent to storage per second:{}".format(measured_fps))
        last_time = current_time
        num_images = 0

# Cleanup
avi.stop()
