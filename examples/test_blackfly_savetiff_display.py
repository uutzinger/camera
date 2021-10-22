##########################################################################
# Testing of display and capture & storage thread combined.
##########################################################################
# Results
# =======
# With Queue: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#  Display Rate: 10
#   Frames captured rate: 524.4
#   Cubes generated rate: 37.4
#   Cubes sent to storage rate: 37.4
#   Frames displayed rate: 9.8
#   Cubes stored rate: 37.4 
#   CPU Usage: 6-7%
#   Disk IO: 195MB/s
#  Display Rate: 30
#   Frames captured rate: 524.4
#   Cubes generated rate: 37.4
#   Cubes sent to storage rate: 37.4
#   Frames displayed rate: 27.4
#   Cubes stored rate: 37.6 
#   CPU Usage: 6-7%
#   Disk IO: 195MB/s
# Without Queue:
#  Display Rate: 10
#   Frames captured rate: 240
#   Cubes generated rate: 2.0
#   Cubes sent to storage rate: 2.0
#   Frames displayed rate: 10
#   Cubes stored rate: 1.8
#   CPU Usage: 5-6%
#   Disk IO: MB/s
##########################################################################
import cv2
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue

looptime = 0.0
use_queue = True
data_cube = np.zeros((14,540,720), dtype=np.uint8)
frame = np.zeros((540,720), dtype=np.uint8)

# Camera configuration file
from configs.blackfly_configs  import configs

# Display
display_interval = 1.0/configs['displayfps']
window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,480)
textLocation1  = (10,520)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up input and output Queue
captureQueue = Queue(maxsize=64)
storageQueue = Queue(maxsize=2)

# Setting up Storage
from camera.streamer.tiffstorageserver import tiffServer
print("Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".tiff"
tiff = tiffServer("C:\\temp\\" + filename)
print("Starting Storage Server")
if use_queue: 
    tiff.start(storageQueue)
else:
    tiff.start()

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
if use_queue:
    camera.start(captureQueue)
else:
    camera.start()

# Initialize Variables
frame_idx = 0               # index to create data cube out of individual frames
num_cubes_sent = 0          # keep track of data cubes sent to storage
num_cubes_generated = 0     # keep track of data cubes generated
last_xps_time = time.time() # keep track of time to dispay performance
last_display  = time.time() # keeo track of time to display images
num_frames_received    = 0  # keep track of how many captured frames reach the main program
num_frames_displayed   = 0  # keep trakc of how many frames are displayed
measured_dps           = 0  # computed in main thread, number of frames displayed per second

# Main Loop
while(True):
    current_time = time.time()

    # wait for new image
    if use_queue:
        (frame_time, frame) = captureQueue.get(block=True, timeout=None)
        data_cube[frame_idx,:,:] = frame
        stat=cv2.sumElems(frame)
        frame_idx += 1
    else:
        if camera.new_frame:
            frame = camera.frame
            frame_time = camera.frame_time
            data_cube[frame_idx,:,:] = frame
            frame_idx += 1

    # When we have a complete dataset:
    if frame_idx >= 14: # 0...13 is populated
        frame_idx = 0
        num_cubes_generated += 1

        if use_queue:
            if not storageQueue.full():
                storageQueue.put((frame_time, data_cube), block=False) 
                num_cubes_sent += 1
            else:
                self.logger.log(logging.DEBUG, "Status:Storage Queue is full!")
        else:
            tiff.framearray_time = frame_time # data array will be saved with this as its name
            tiff.framearray = data_cube # transfer data array to storage server
            num_cubes_sent += 1

    # Display performance in main loop
    if current_time - last_xps_time >= 5.0:
        # how many data cubes did we create
        measured_cps_generated = num_cubes_generated/5.0
        logger.log(logging.DEBUG, "Status:captured cubes generated per second:{}".format(measured_cps_generated))
        num_cubes_generated = 0
        # how many data cubes did we send to storage
        measured_cps_sent = num_cubes_sent/5.0
        logger.log(logging.DEBUG, "Status:cubes sent to storage per second:{}".format(measured_cps_sent))
        num_cubes_sent = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_xps_time = current_time

    if (current_time - last_display) >= display_interval:
        display_frame = frame.copy()
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, display_frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
            break
        last_display = current_time
        num_frames_displayed += 1

    if not use_queue:
        # make sure the while-loop takes at least looptime to complete, otherwise CPU use is high in main thread and capture and storage is slow
        # since queue is blocking, we dont need this when we use queue
        delay_time = looptime - (time.time() - current_time) 
        if  delay_time >= 0.001:
            time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

# Cleanup
camera.stop()
tiff.stop()
cv2.destroyAllWindows()
