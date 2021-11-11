##########################################################################
# Testing of display and capture & storage thread combined
# Scan for camera
# Save mkv files
##########################################################################
# 7-9% CPU usage
##########################################################################
import cv2
import logging
import time
import numpy as np
from datetime import datetime
from queue import Queue
import platform

# Probe the cameras, return indeces, fourcc, default resolution
def probeCameras():
    # check for up to 10 cameras
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            tmp = cap.get(cv2.CAP_PROP_FOURCC)
            fourcc = "".join([chr((int(tmp) >> 8 * i) & 0xFF) for i in range(4)])
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            arr.append({"index": index, "fourcc": fourcc, "width": width, "height": height})
        index += 1
        i -= 1
    return arr

# Camera Signature
# In case we have multiple cameras, we can search for default driver settings
# and compare to camera signature, opencv unfortunately does not return the 
# serial number of the camera
# Example: Generic Webcam: 640, 480, YUYV
# Example: FLIRLepton: 160, 120, BGR3
widthSig = 640
heightSig = 480
#fourccSig = "YUYV"
fourccSig = "\x16\x00\x00\x00"
# default camera starts at 0 by operating system
camera_index = 0

# Scan all camera
camprops = probeCameras()
# Try to find the one that matches our signature
score = 0
for i in range(len(camprops)):
    try: found_fourcc = 1 if camprops[i]['fourcc'] == fourccSig else 0            
    except: found_fourcc = 0
    try: found_width = 1  if camprops[i]['width']  == widthSig  else 0
    except: found_width =  0
    try: found_height = 1 if camprops[i]['height'] == heightSig else 0   
    except: found_height = 0
    tmp = found_fourcc+found_width+found_height
    if tmp > score:
        score = tmp
        camera_index = i

# Camera configuration file
# -Dell Inspiron 15 internal camer
# from configs.dell_internal_configs  import configs as configs
# -Eluktronics Max-15 internal camera
from configs.eluk_configs import configs as configs
# -Generic webcam
#from configs.generic_1080p import configs as configs
# -Nano Jetson IMX219 camera
# from configs.nano_IMX219_configs  import configs as configs
# -Raspberry Pi v1 & v2 camera
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs
# -ELP MAX15 internal camera
# from configs.ELP1080p_configs  import configs as configs
# -FLIR Lepton 3.5
# from configs.FLIRlepton35 import confgis as configs

# Display
display_interval = 1.0/configs['displayfps']
window_name      = 'Camera'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,60)
textLocation2    = (10,100)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Setting up input and output Queue
captureQueue = Queue(maxsize=64)
storageQueue = Queue(maxsize=64)
# here add additional queues if you have more than one camera. 
# Each camera needs capture and saving queue

# Setting up Storage
from camera.streamer.mkvstorageserver import mkvServer
print("Starting Storage Server")
now = datetime.now()
fps  = configs['fps']
size = configs['camera_res']
filename = now.strftime("%Y%m%d%H%M%S") + ".mkv"
mkv = mkvServer("C:\\temp\\" + filename, fps, size)
print("Starting Storage Server")
mkv.start(storageQueue)

# Create camera interface
print("Starting Capture")

# Create camera interface based on computer OS you are running
# plat can be Windows, Linux, MaxOS
plat = platform.system()
if plat == 'Linux':
    if platform.machine() == "aarch64": # this is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is raspberry for me
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs, camera_index)
else:
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs, camera_index)
print("Getting Images")
camera.start(captureQueue)

print("Getting Images")
camera.start(captureQueue)

# Initialize Variables
num_frames_sent      = 0           # keep track of data cubes sent to storage
last_xps_time        = time.time() # keep track of time to dispay performance
last_display         = time.time() # keeo track of time to display images
num_frames_displayed = 0           # keep trakc of how many frames are displayed
measured_dps         = 0           # computed in main thread, number of frames displayed per second

# Main Loop
while(True):
    current_time = time.time()

    # wait for new image
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)
    # if you have two cameras with different fps settings, we need to figure out here how to 
    # obtain images in non blocking fashion as the slowest would prevail and buffer over runs on faster
    
    if not storageQueue.full():
        storageQueue.put((frame_time, frame), block=False) 
        num_frames_sent += 1
    else:
        logger.log(logging.WARNING, "Status:Storage Queue is full!")

    # Display performance in main loop
    if current_time - last_xps_time >= 5.0:
        # how many frames did we send to storage
        measured_fps_sent = num_frames_sent/5.0
        logger.log(logging.INFO, "Status:frames sent to storage per second:{}".format(measured_fps_sent))
        num_frames_sent = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/5.0
        logger.log(logging.INFO, "Status:frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_xps_time = current_time

    if (current_time - last_display) >= display_interval:
        frame_display = frame.copy()
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(frame_display,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display,"Storage FPS:{} [Hz]".format(mkv.measured_cps),    textLocation2, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame_display)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): # this likely is the reason that display frame rate is not faster than 60fps.
            break
        last_display = current_time
        num_frames_displayed += 1

# Cleanup
camera.stop()
mkv.stop()
cv2.destroyAllWindows()
