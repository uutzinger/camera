##########################################################################
# Testing of display and capture & storage thread combined.
##########################################################################
#
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
from camera.configs.BEN_configs  import configs

# Display
display_interval = 1.0/configs['displayfps']
window_name0   = 'Camera0'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,440)
textLocation1  = (10,480)
textLocation2  = (10,520)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name0, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up input and output Queue
captureQueue0 = Queue(maxsize=64)
storageQueue0 = Queue(maxsize=2)
# here add additional queues if you have more than one camera. 
# Each camera needs capture and saving queue

# Setting up Storage
from camera.streamer.storageserver import aviServer
print("Starting Storage Server")
now = datetime.now()
fps  = configs['fps']
size = configs['camera_res']
filename0 = now.strftime("%Y%m%d%H%M%S") + "_0.avi"
avi0 = aviServer("C:\\temp\\" + filename0, fps, size)
# create multiple avi1 = ... if you have multiple avis to stream to
print("Starting Storage Server")
if use_queue: 
    avi0.start(storageQueue0)
else:
    avi0.start()

# Create camera interface
print("Starting Capture")
# you will need to create camera0 = ... for each camera
# Create camera interface based on computer OS you are running

plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs,0) # 0 is camera number
elif plat == 'Linux':
    if platform.machine() == "aarch64":  # This is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera0 = nanoCapture(configs,0) # 0 is camera number
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is reaspberry pi for me
        from camera.capture.cv2capture import cv2Capture
        camera0 = cv2Capture(configs,0) # 0 is camera number
        # from picapture import piCapture
        # camera = piCapture()
elif plat == 'MacOS': # dont have
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs,0) # 0 is camera number
else:
    from camera.capture.cv2capture import cv2Capture
    camera0 = cv2Capture(configs,0) # 0 is camera number

print("Getting Images")
if use_queue:
    camera0.start(captureQueue0)
else:
    camera0.start()

# Initialize Variables
num_frames0_sent      = 0          # keep track of data cubes sent to storage
num_frames0_generated = 0          # keep track of data cubes generated
last_xps_time        = time.time() # keep track of time to dispay performance
last_display         = time.time() # keeo track of time to display images
num_frames0_received  = 0          # keep track of how many captured frames reach the main program
num_frames0_displayed = 0          # keep trakc of how many frames are displayed
measured_dps         = 0           # computed in main thread, number of frames displayed per second

# Main Loop
while(True):
    current_time = time.time()

    # wait for new image
    if use_queue:
        (frame0_time, frame0) = captureQueue0.get(block=True, timeout=None)
    else:
        if camera0.new_frame:
            frame0 = camera0.frame
            frame0_time = camera0.frame_time

    if use_queue:
        if not storageQueue0.full():
            storageQueue0.put((frame0_time, frame0), block=False) 
            num_frames0_sent += 1
        else:
            self.logger.log(logging.DEBUG, "Status:0:Storage Queue is full!")
    else:
        avi0.framearray = frame0 # transfer data array to storage server
        num_frames0_sent += 1

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
        measured_dps = num_frames0_displayed/5.0
        logger.log(logging.DEBUG, "Status:0:frames displayed per second:{}".format(measured_dps))
        num_frames0_displayed = 0
        last_xps_time = current_time

    if (current_time - last_display) > display_interval:
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(frame0,"Capture FPS:{} [Hz]".format(camera0.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame0,"Display FPS:{} [Hz]".format(measured_dps),         textLocation1, font, fontScale, fontColor, lineType)
        cv2.putText(frame0,"Storage FPS:{} [Hz]".format(avi0.measured_cps),    textLocation2, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name0, frame0)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
            break
        last_display = current_time
        num_frames0_displayed += 1

    if not use_queue:
        # make sure the while-loop takes at least looptime to complete, otherwise CPU use is high in main thread and capture and storage is slow
        # since queue is blocking, we dont need this when we use queue
        delay_time = looptime - (time.time() - current_time) 
        if  delay_time >= 0.001:
            time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

# Cleanup
camera0.stop()
avi0.stop()
cv2.destroyAllWindows()
