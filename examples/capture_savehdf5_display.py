##########################################################################
# Testing of display and capture & storage thread combined.
# Scan for camera
# Aquire 14 images
# Convert to b/w
# Save hdf5 files
##########################################################################
# Results
#
#
##########################################################################

# System
import logging, time, platform
from datetime import datetime

# Matrix Algebra
import numpy as np
from numba import vectorize

# OpenCV
import cv2

configs = {
    'camera_res'      : (1280, 720 ),   # width & height
    'exposure'        : -6,             # -1,0 = auto, 1...max=frame interval, 
    'autoexposure'    : 1.0,            # depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 30,             # 15, 30, 40, 90, 120, 180
    'fourcc'          : -1,             # n.a.
    'buffersize'      : -1,             # n.a.
    'output_res'      : (-1, -1),       # Output resolution, -1,-1 no change
    'flip'            : 0,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'displayfps'       : 30             # frame rate for display server
}

if configs['displayfps'] >= configs['fps']: 
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

res = configs['camera_res']
height = res[1]
width = res[0]
measure_time = 5.0 # average measurements over 5 secs
camera_index = 0 # default camera starts at 0 by operating system


# Processing
data_cube      = np.zeros((14, height, width), 'uint8')
background     = np.zeros((height, width), 'uint8')                         # where we keep bg
flatfield      = np.cast['uint16'](2**8.*np.random.random((height, width))) # flatfield correction scaled so that 255=100%
inten         = np.zeros(14, 'uint16')                                # helper to find background image
data_cube_corr = np.zeros((14, height, width), 'uint16')                    # resulting data cube on CPU
# Numpy Vectorized
@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def vector_np(data_cube, background, flatfield):
    return np.multiply(np.subtract(data_cube, background), flatfield) # 16bit multiplication

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up storage
from camera.streamer.h5storageserver import h5Server
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("C:\\temp\\" + filename)
logger.log(logging.INFO, "Starting Storage Server")
hdf5.start()

# Create camera interface
logger.log(logging.INFO, "Starting Capture")
# Computer OS and platform dependent
plat = platform.system()
if plat == 'Linux' and platform.machine() == "aarch64": # this is jetson nano for me
    from camera.capture.nanocapture import nanoCapture
    camera = nanoCapture(configs, camera_index)
else:
    from camera.capture.cv2capture_process import cv2Capture
    camera = cv2Capture(configs, camera_index)
logger.log(logging.INFO, "Getting Images")
camera.start()

# Display
window_name    = 'Main'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,height-40)
textLocation1  = (10,height-20)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Initialize Variables
frame_idx              = 0  # index to create data cube out of individual frames
num_cubes_stored       = 0  # keep track of data cubes sent to storage
num_cubes_generated    = 0  # keep track of data cubes generated
last_time              = time.perf_counter() # keep track of time to dispay performance
last_display           = time.perf_counter() # keeo track of time to display images
num_frames_received    = 0  # keep track of how many captured frames reach the main program
num_frames_displayed   = 0  # keep track of how many frames are displayed
measured_dps           = 0  # computed in main thread, number of frames displayed per second

# Main Loop
stop =  False
while(not stop):
    current_time = time.perf_counter()

    # Camera
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    data_cube[frame_idx,:,:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_idx += 1
    while not camera.log.empty():
        (level, msg)=camera.log.get_nowait()
        logger.log(level, "Status:{}".format(msg))

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_generated += 1

        # HDF5 
        try: 
            hdf5.queue.put_nowait((frame_time, data_cube)) 
            num_cubes_stored += 1
        except:
            logger.log(logging.DEBUG, "Status:Storage Queue is full!")

        # Background and Field Correction
        # Where is my background?
        _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
        frame_idx_bg = np.argmin(inten)  # minimum intensity is in this frame
        background = data_cube[frame_idx_bg, :, :]
        # Correct the data
        vector_np(data_cube, flatfield, background, out = data_cube_corr)

    while not hdf5.log.empty():
        (level, msg)=hdf5.log.get_nowait()
        logger.log(level, "Status:{}".format(msg))

    # Display performance in main loop
    if current_time - last_time >= measure_time:
        # how many data cubes did we create
        measured_cps_generated = num_cubes_generated/measure_time
        logger.log(logging.DEBUG, "Status:captured cubes generated per second:{}".format(measured_cps_generated))
        num_cubes_generated = 0
        # how many data cubes did we send to storage
        measured_cps_stored = num_cubes_stored/measure_time
        logger.log(logging.DEBUG, "Status:cubes sent to storage per second:{}".format(measured_cps_stored))
        num_cubes_sent = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/measure_time
        logger.log(logging.DEBUG, "Status:frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_time = current_time

    if (current_time - last_display) >= display_interval:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True
        last_display = current_time
        num_frames_displayed += 1

# Cleanup
camera.stop()
hdf5.stop()
cv2.destroyAllWindows()
