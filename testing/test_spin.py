##########################################################################
# Testing of spin capture thread.
# No display of images
# A data cube has 14 images of 720x540 and is 5.5MBytes in size.
# The camera should achieve 525 frames per second.
# The combining 14 frames into a datacube should occur 37.5 times per second
##########################################################################
# Results
# =======
#   37.4 cubes per second
#   524.2 frames per second
#   CPU Usage: 6-7%
##########################################################################
import logging
import time
import numpy as np

# Camera configuration file
from configs.spin_configs  import configs

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("spinmain")

# Create camera interface
from camera.capture.spincapture import spinCapture
print("Starting Capture")
camera = spinCapture(configs)
print("Getting Images")
camera.start()

capture_fps = float(configs.get("fps", 0) or 0)
loop_interval = 1.0 / (2.0 * (capture_fps if capture_fps > 0 else 30.0))

depth = 14
data_cube = None

# Initialize Variables
last_cps_time = time.perf_counter()
measured_cps  = 0
frame_idx     = 0 
num_cubes     = 0

while True:
    current_time = time.perf_counter()

    if camera.buffer and camera.buffer.avail > 0:
        frame, _ts = camera.buffer.pull(copy=False)
    else:
        frame = None

    try:
        while not camera.log.empty():
            level, msg = camera.log.get_nowait()
            logger.log(level, "%s", msg)
    except Exception:
        pass

    if frame is None:
        remaining = loop_interval - (time.perf_counter() - current_time)
        if remaining > 0:
            time.sleep(remaining)
        continue

    if data_cube is None:
        data_cube = np.zeros((depth, *frame.shape), dtype=frame.dtype)

    data_cube[frame_idx, ...] = frame
    frame_idx += 1

    # When we have a complete dataset:
    if frame_idx >= depth: # 0...depth-1 is populated
        frame_idx = 0
        num_cubes += 1

    if current_time - last_cps_time >= 5.0:
        measured_cps = num_cubes/5.0
        logger.log(logging.DEBUG, "Status:Cubes received per second:{}".format(measured_cps))
        last_cps_time = current_time
        num_cubes = 0

    remaining = loop_interval - (time.perf_counter() - current_time)
    if remaining > 0:
        time.sleep(remaining)

camera.stop()
camera.join(timeout=2.0)
if hasattr(camera, "close"):
    camera.close()
