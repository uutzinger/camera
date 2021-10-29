##########################################################################
# Testing of display and capture & storage thread combined
# if you want to expand to multiple cameras, you can duplicate any 
# variable and object ending with 0 to one with 1
# Main issue at this time is if one camera has lower fps than the other
##########################################################################
#
##########################################################################
import cv2
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue
import platform

# Camera configuration file
from configs.eluk_configs import configs as configs0

# Display
display0_interval = 1.0/configs0['displayfps']
window_name0     = 'Camera0'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation00   = (10,20)
textLocation10   = (10,60)
textLocation20   = (10,100)
fontScale0       = 1
fontColor        = (255,255,255)
lineType         = 2
cv2.namedWindow(window_name0, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up input and output Queue
captureQueue0 = Queue(maxsize=64)
storageQueue0 = Queue(maxsize=64)
# here add additional queues if you have more than one camera. 
# Each camera needs capture and saving queue

# Setting up Storage
from camera.streamer.avistorageserver import aviServer
print("Starting Storage Server")
now = datetime.now()
fps0  = configs0['fps']
size0 = configs0['camera_res']
filename0 = now.strftime("%Y%m%d%H%M%S") + "_0.avi"
avi0 = aviServer("C:\\temp\\" + filename0, fps0, size0)
# create multiple avi1 = ... if you have multiple avis to stream to
print("Starting Storage Server")
avi0.start(storageQueue0)

# Create camera interface
print("Starting Capture")
# you will need to create camera0 = ... for each camera
# Create camera interface based on computer OS you are running

plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs0,0) # 0 is camera number
elif plat == 'Linux':
    if platform.machine() == "aarch64":  # This is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera0 = nanoCapture(configs0,0) # 0 is camera number
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is reaspberry pi for me
        from camera.capture.cv2capture import cv2Capture
        camera0 = cv2Capture(configs0,0) # 0 is camera number
elif plat == 'MacOS': # 
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs0,0) # 0 is camera number
else:
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs0,0) # 0 is camera number

print("Getting Images")
camera0.start(captureQueue0)

# Initialize Variables
num_frames0_sent      = 0          # keep track of data cubes sent to storage
num_frames0_generated = 0          # keep track of data cubes generated
last_xps_time        = time.time() # keep track of time to dispay performance
last_display0        = time.time() # keeo track of time to display images
num_frames0_received  = 0          # keep track of how many captured frames reach the main program
num_frames0_displayed = 0          # keep trakc of how many frames are displayed
measured_dps         = 0           # computed in main thread, number of frames displayed per second

# Main Loop
while(True):
    current_time = time.time()

    # wait for new image
    (frame0_time, frame0) = captureQueue0.get(block=True, timeout=None)
    # if you have two cameras with different fps settings, we need to figure out here how to 
    # obtain images in non blocking fashion as the slowest would prevail and buffer over runs on faster
    
    if not storageQueue0.full():
        storageQueue0.put((frame0_time, frame0), block=False) 
        num_frames0_sent += 1
    else:
        logger.log(logging.DEBUG, "Status:0:Storage Queue is full!")

    # Display performance in main loop
    if current_time - last_xps_time >= 5.0:
        # how many data cubes did we create
        measured_fps_generated = num_frames0_generated/5.0
        logger.log(logging.DEBUG, "Status:0:captured frames generated per second:{}".format(measured_fps_generated))
        num_frames0_generated = 0
        # how many data cubes did we send to storage
        measured_fps_sent = num_frames0_sent/5.0
        logger.log(logging.DEBUG, "Status:0:frames sent to storage per second:{}".format(measured_fps_sent))
        num_frames_sent = 0
        # how many frames did we display
        measured0_dps = num_frames0_displayed/5.0
        logger.log(logging.DEBUG, "Status:0:frames displayed per second:{}".format(measured0_dps))
        num_frames0_displayed = 0
        last_xps_time = current_time

    if (current_time - last_display0) >= display0_interval:
        frame0_display = frame0.copy()
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(frame0_display,"Capture FPS:{} [Hz]".format(camera0.measured_fps), textLocation00, font, fontScale0, fontColor, lineType)
        cv2.putText(frame0_display,"Display FPS:{} [Hz]".format(measured0_dps),         textLocation10, font, fontScale0, fontColor, lineType)
        cv2.putText(frame0_display,"Storage FPS:{} [Hz]".format(avi0.measured_cps),    textLocation20, font, fontScale0, fontColor, lineType)
        cv2.imshow(window_name0, frame0_display)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
            break
        last_display0 = current_time
        num_frames0_displayed += 1

# Cleanup
camera0.stop()
avi0.stop()
cv2.destroyAllWindows()
