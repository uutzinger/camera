##########################################################################
# Testing of display and capture 
# Optional scan for camera
##########################################################################
# 2% CPU usage
##########################################################################

import cv2
import logging
import time
import platform
from queue import Queue
import numpy as np

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

loop_interval = 1.0/200.0

# Camera Signature
# In case we have multiple camera, we can search for default driver settings
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
# # Scan all camera
# camprops = probeCameras()
# # Try to find the one that matches our signature
# score = 0
# for i in range(len(camprops)):
#     try: found_fourcc = 1 if camprops[i]['fourcc'] == fourccSig else 0            
#     except: found_fourcc = 0
#     try: found_width = 1  if camprops[i]['width']  == widthSig  else 0
#     except: found_width =  0
#     try: found_height = 1 if camprops[i]['height'] == heightSig else 0   
#     except: found_height = 0
#     tmp = found_fourcc+found_width+found_height
#     if tmp > score:
#         score = tmp
#         camera_index = i

# -Dell Inspiron 15 internal camer
# from configs.dell_internal_configs  import configs as configs
# -Eluktronics Max-15 internal camera
from configs.eluk_configs import configs as configs
# -Generic webcam
# from configs.generic_1080p import configs as configs
# -Nano Jetson IMX219 camera
# from configs.nano_IMX219_configs  import configs as configs
# -Raspberry Pi v1 & v2 camera
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs
# -ELP MAX15 internal camera
# from configs.ELP1080p_configs  import configs as configs
# -FLIR Lepton 3.5
# from configs.FLIRlepton35 import confgis as configs

if configs['displayfps'] >= configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

dps_measure_time = 5.0 # average measurements over 5 secs

window_name      = 'Camera'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,60)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("CV2Capture")

# Setting up input and/or output Queue
# Increase maxsize if you get queue is full statements
captureQueue = Queue(maxsize=32)

configs['output_res']

# Create camera interface based on computer OS you are running
plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs, camera_index)
elif plat == 'Linux':
    if platform.machine() == "aarch64": # for me this is jetson nano
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is raspberry for me
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs, camera_index)
elif plat == 'MacOS':
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs, camera_index)
else:
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)

print("Getting Images")
camera.start(captureQueue)

# Initialize Variables
last_display   = time.time()
last_fps_time  = time.time()
measured_dps   = 0
num_frames_received    = 0
num_frames_displayed   = 0

while(cv2.getWindowProperty(window_name, 0) >= 0):
    current_time = time.time()
    #start_time   = time.perf_counter()

    # wait for new image
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)
    num_frames_received += 1

    if current_time - last_fps_time >= dps_measure_time:
        measured_fps = num_frames_received/dps_measure_time
        logger.log(logging.INFO, "Status:Frames received per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/dps_measure_time
        logger.log(logging.INFO, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

    if (current_time - last_display) >= display_interval:
        frame_display = frame.copy()        
        cv2.putText(frame_display,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame_display)
        # quit the program if users enter q or closes the display window
        # the waitKey function limits the display frame rate to about 30fps for me
        # without waitKey the opencv window is not refreshed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_display = current_time
        num_frames_displayed += 1

    # avoid looping unnecessarely, 
    # this is only relevant for low fps
    #end_time = time.perf_counter()
    #delay_time = loop_interval - (end_time - start_time)
    #if  delay_time >= 0.005:
    #    time.sleep(delay_time)  # this creates at least 3ms delay, regardless of delay_time

camera.stop()
cv2.destroyAllWindows()
