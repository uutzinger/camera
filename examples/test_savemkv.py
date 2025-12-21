##########################################################################
# Testing of storage server thread.
# No camera involved
##########################################################################
# Results
#   450-500 frames per second
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
filename = now.strftime("%Y%m%d%H%M%S") + ".mkv"
from camera.streamer.mkvstorageserver import mkvServer
logger.log(logging.INFO, "Setting up Storage Server")
mkv = mkvServer("C:\\temp\\" + filename, fps, size)
logger.log(logging.INFO, "Starting Storage Server")
mkv.start()

num_frames = 0 
last_time = time.time()

# Main Loop
while True:
    current_time = time.time()

    mkv.queue.put((current_time, image), block=True, timeout=None) 
    num_frames += 1

    if current_time - last_time >= 5.0:
        measured_cps = num_frames/5.0
        logger.log(logging.INFO, "Status:Frames sent to storage per second:{}".format(measured_cps))
        last_time = current_time
        num_frames = 0

    while not mkv.log.empty():
        (level, msg)=mkv.log.get_nowait()
        logger.log(level, "Status:{}".format(msg))

# Cleanup
mkv.stop()
