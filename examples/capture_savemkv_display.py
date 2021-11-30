##########################################################################
# Testing of display and capture & storage thread combined
# Scan for camera
# Save mkv files
##########################################################################
# Results
# 1080p 30fps capture, 30fps save, 30fps display
# 4% CPU usage
##########################################################################

# OpenCV
import cv2

# System
import logging, time, platform
from datetime import datetime
from camera.utils import probeCameras

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
camprops = probeCameras(10)
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

# Setting up Storage
from camera.streamer.mkvstorageserver import mkvServer
logger.log(logging.INFO, "Starting Storage Server")
now = datetime.now()
fps  = configs['fps']
size = configs['camera_res']
filename = now.strftime("%Y%m%d%H%M%S") + ".mkv"
mkv = mkvServer("C:\\temp\\" + filename, fps, size)
logger.log(logging.INFO, "Starting Storage Server")
mkv.start()

# Create camera interface
logger.log(logging.INFO, "Starting Capture")
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
logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
num_frames_sent      = 0                   # keep track of data cubes sent to storage
last_time            = time.perf_counter() # keep track of time to dispay performance
last_display         = time.perf_counter() # keeo track of time to display images
num_frames_displayed = 0                   # keep trakc of how many frames are displayed
measured_dps         = 0                   # computed in main thread, number of frames displayed per second

# Main Loop
stop=False
while(not stop):
    current_time = time.perf_counter()

    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, "{}".format(msg))

    if not mkv.queue.full():
        mkv.queue.put_nowait((frame_time, frame)) 
        num_frames_sent += 1
    else:
        logger.log(logging.WARNING, "Status:Storage Queue is full!")
    while not mkv.log.empty():
        (level, msg)=mkv.log.get_nowait()
        logger.log(level, "{}".format(msg))

    # Performances in main loop
    if (current_time - last_time) >= 5.0:
        # how many frames did we send to storage
        measured_fps_sent = num_frames_sent/5.0
        logger.log(logging.INFO, "Status:frames sent to storage per second:{}".format(measured_fps_sent))
        num_frames_sent = 0
        # how many frames did we display
        measured_dps = num_frames_displayed/5.0
        logger.log(logging.INFO, "Status:frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_time = current_time

    # if you want to run at full speed remove the if statement
    if (current_time - last_display) >= 0.8*display_interval:
        frame_display = frame.copy()
        # This section creates significant delay and we need to throttle the display to maintain max capture and storage rate
        cv2.putText(frame_display,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.putText(frame_display,"Storage FPS:{} [Hz]".format(mkv.measured_cps),    textLocation2, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame_display)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True
        last_display = current_time
        num_frames_displayed += 1

# Cleanup
camera.stop()
mkv.stop()
cv2.destroyAllWindows()
