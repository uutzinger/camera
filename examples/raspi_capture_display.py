##########################################################################
# Testing of display and capture 
# Optional scan for camera
##########################################################################
# 2% CPU usage
##########################################################################

import cv2
import logging
import time

from queue import Empty

loop_interval = 1.0/200.0

# default camera starts at 0 by operating system
camera_index = 0

# Can import or define camera configuration below
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs

configs = {
    ##############################################
    # Camera Settings
    # 320x240 90fps
    # 640x480 90fps
    # 1280x720 60fps
    # 1920x1080 6.4fps
    # 2592x1944 6.4fps
    ##############################################
    'camera_res'      : (320, 240),     # camera width & height
    'exposure'        : 1000,          # microseconds
    'autoexposure'    : 0,              # 
    'fps'             : 120,             # 
    'fourcc'          : 'YU12',         # 
    'buffersize'      : 4,              # default is 4 for V4L2, max 10, 
    'output_res'      : (-1, -1),       # Output resolution 
    'flip'            : 0,              # 0=norotation 
    'displayfps'      : 10              # frame rate for display server
    }

if configs['displayfps'] >= 0.8*configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

dps_measure_time = 5.0 # assess performance every 5 secs

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
logger = logging.getLogger("Raspi Capture")

# Create camera interface based on computer OS you are running
# Prefer Raspberry Pi Picamera2/libcamera when available, otherwise fall back to OpenCV.
camera = None
try:
    from camera.capture.picamera2capture import piCamera2Capture

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not getattr(camera, 'cam_open', False):
        camera = None
except Exception:
    camera = None

if camera is None:
    from camera.capture.cv2capture import cv2Capture

    camera = cv2Capture(configs, camera_index)

logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
last_display   = time.perf_counter()
last_fps_time  = time.perf_counter()
measured_dps   = 0
num_frames_received    = 0
num_frames_displayed   = 0

stop = False
while(not stop):

    current_time = time.perf_counter()

    # wait for new image (timeout keeps UI responsive even if capture stalls)
    try:
        (frame_time, frame) = camera.capture.get(timeout=0.25)
        num_frames_received += 1
    except Empty:
        frame = None

    #display log
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, "{}".format(msg))

    # calc stats
    if (current_time - last_fps_time) >= dps_measure_time:
        measured_fps = num_frames_received/dps_measure_time
        logger.log(logging.INFO, "MAIN:Frames received per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/dps_measure_time
        logger.log(logging.INFO, "MAIN:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

    # display
    if (frame is not None) and ((current_time - last_display) >= display_interval):
        frame_display = frame.copy()        
        cv2.putText(frame_display,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame_display)
        ## quit the program if users enter q or closes the display window
        ## the waitKey function limits the display frame rate
        ## without waitKey the opencv window is not refreshed
        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True
        try:
            if cv2.getWindowProperty(window_name, 0) < 0: stop = True
        except: stop =True
        last_display = current_time
        num_frames_displayed += 1

    # avoid looping unnecessarely, this is only relevant for low fps and non blocking capture
    #end_time = time.perf_counter()
    #delay_time = loop_interval - (end_time - current_time)
    #if  delay_time >= 0.005:
    #    time.sleep(delay_time)  # this creates at least 3ms delay, regardless of delay_time

# Clean up
camera.stop()
try:
    camera.join(timeout=2.0)
except Exception:
    pass
try:
    camera.close_cam()
except Exception:
    pass
cv2.destroyAllWindows()
