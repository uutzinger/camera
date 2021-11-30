##########################################################################
# Testing of display and capture & storage thread combined.
##########################################################################
# Results
##########################################################################

# System
import logging, time
from datetime import datetime
from timeit import default_timer as timer

# Matrix Algebra
import numpy as np
from numba import vectorize

# OpenCV
import cv2

# Camera configuration file
from examples.configs.blackfly_configs  import configs

if configs['displayfps'] >= configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

dps_measure_time = 5.0 # average measurements over 5 secs

res = configs['camera_res']
height = res[1]
width = res[0]
measure_time = 5.0 # average measurements over 5 secs
camera_index = 0 # default camera starts at 0 by operating system

# Processing
data_cube      = np.zeros((14, height, width), 'uint8')                      # allocate space for the 14 images
background     = np.zeros((height, width), 'uint8')                          # allocate space for background image
flatfield      = np.cast['uint16'](2**8.*np.ones((height, width), 'uint16')) # flatfield correction image, scaled so that 255=100%
inten          = np.zeros(14, 'uint16')                                      # helper to find background image in the 14 images
data_cube_corr = np.zeros((14, height, width), 'uint16')                     # allocated space for the corrected 14 images

# Calibration function Numpy vectorized
@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def calibrate(data_cube, background, flatfield):
    return np.multiply(np.subtract(data_cube, background), flatfield) # 16bit multiplication

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up Storage
from camera.streamer.h5storageserver import h5Server
logger.log(logging.INFO, "Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("D:\\temp\\" + filename)
print("Starting Storage Server")
hdf5.start()

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
logger.log(logging.INFO, "Starting Capture")
camera = blackflyCapture(configs)
while not camera.log.empty():
    (level, msg)=camera.log.get_nowait()
    logger.log(level, msg)
logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
frame_idx              = 0  # index to create data cube out of individual frames
num_cubes_stored       = 0  # keep track of data cubes sent to storage
num_cubes_generated    = 0  # keep track of data cubes generated
last_time              = time.perf_counter() # keep track of time to dispay performance
last_display           = time.perf_counter() # keeo track of time to display images
num_frames_received    = 0  # keep track of how many captured frames reach the main program
num_frames_displayed   = 0  # keep trakc of how many frames are displayed
measured_dps           = 0  # computed in main thread, number of frames displayed per second
proc_time              = 0 

# Display
window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,480)
textLocation1  = (10,520)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

cv2.namedWindow('Proc', cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Main Loop
stop = False
while(not stop):
    current_time = time.perf_counter()

    # Camera
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    data_cube[frame_idx,:,:] = frame
    frame_idx += 1
    while not camera.log.empty():
        (level, msg)=camera.log.get_nowait()
        logger.log(level, msg)

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_generated += 1

        # Background and Field Correction
        # a) where is my background?
        proc_start_time = timer()
        _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
        frame_idx_bg = np.argmin(inten)  # minimum intensity is in this frame
        background = data_cube[frame_idx_bg, :, :]
        # b) correct the data
        calibrate(data_cube, flatfield, background, out = data_cube_corr)
        proc_end_time = timer()
        proc_time += (proc_end_time-proc_start_time)

        # HDF5 
        try: 
            hdf5.queue.put_nowait((frame_time, data_cube_corr)) 
            num_cubes_stored += 1 # executed if above was successful
        except:
            pass
            # logger.log(logging.WARNING, "HDF5:Storage Queue is full!")

    while not hdf5.log.empty():
        (level, msg)=hdf5.log.get_nowait()
        logger.log(level, msg)

    # Display performance in main loop
    if current_time - last_time >= measure_time:
        # how much time did it take to process the data
        if num_cubes_generated > 0:
            logger.log(logging.INFO, "Status:process time:{:.2f}ms".format(proc_time*1000./num_cubes_generated))
        # how many data cubes did we create
        measured_cps_generated = num_cubes_generated/measure_time
        logger.log(logging.INFO, "Status:captured cubes generated per second:{}".format(measured_cps_generated))
        num_cubes_generated = 0
        # how many data cubes did we send to storage
        measured_cps_stored = num_cubes_stored/measure_time
        logger.log(logging.INFO, "Status:cubes sent to storage per second:{}".format(measured_cps_stored))
        num_cubes_stored = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/measure_time
        logger.log(logging.INFO, "Status:frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_time = current_time

    if (current_time - last_display) >= display_interval:
        display_frame = np.cast['uint8'](data_cube_corr[0,:,:].copy()/(2**8))
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(display_frame,"Store CPS:{} [Hz]".format(hdf5.measured_cps), textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, display_frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True # this likely is the reason that display frame rate is not faster than 60fps.
        last_display = current_time
        num_frames_displayed += 1

# Cleanup
camera.stop()
hdf5.stop()
cv2.destroyAllWindows()
