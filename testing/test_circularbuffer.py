##########################################################################
# Testing of circular buffer
# No camera involved
##########################################################################
# Results
# =======
# 80-100/sec (includes randomnumber generated datacube)
# 0.5ms per append
##########################################################################
import collections
import logging
import time
import numpy as np
from datetime import datetime

circular_buffer = collections.deque(maxlen=100)

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Storage")

# Setting up Storage
now = datetime.now()
append_time = 0.

num_cubes = 0 
last_cps_time = time.time()
cube_time = 0

# Main Loop
while True:
    current_time = time.time()
    data_cube = np.random.randint(0, 255, (14, 540, 720), 'uint8') 
    # data_cube = np.zeros((14, 540, 720), 'uint8')
    start_time = time.perf_counter()
    circular_buffer.append(data_cube)
    end_time = time.perf_counter()
    append_time=append_time + (end_time-start_time)
    num_cubes += 1

    if current_time - last_cps_time >= 1.0:
        measured_cps = num_cubes/1.0
        logger.log(logging.DEBUG, "Status:Cubes sent to storage per second:{}".format(measured_cps))
        logger.log(logging.DEBUG, "Status:Append time:{}".format(append_time/num_cubes))
        last_cps_time = current_time
        num_cubes = 0
        append_time = 0.

# Cleanup
