##########################################################################
# Testing of capture to rtp 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging
import platform
import time
from queue import Queue

# default camera starts at 0 by operating system
camera_index = 0

window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation   = (10,20)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# -Dell Inspiron 15 internal camer
# from configs.dell_internal_configs  import configs as configs
# -Eluktronics Max-15 internal camera
from eluk_configs import configs as configs
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
    rtp_interval = 1.0/configs['fps']
    rtp_fps = configs['fps']
else:
    rtp_interval = 1.0/configs['displayfps']
    rtp_fps = configs['displayfps']

rtp_size = configs['output_res']
if rtp_size[0]<=0 or rtp_size[1]<=0: 
    rtp_size =  configs['camera_res']

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Capture2rtp")

# Setting up input and/or output Queue
# Increase maxsize if you get queue is full statements
captureQueue = Queue(maxsize=32)
rtpQueue     = Queue(maxsize=32)

# Setting up Storage
from camera.streamer.rtpserver import rtpServer
print("Starting rtp Server")
rtp = rtpServer(resolution = rtp_size, fps=rtp_fps, host='127.0.0.1', port=554, bitrate=2048, GPU=False)
rtp.start(rtpQueue)

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

# Initialize Variables
last_rtp = time.time()

while True:
    current_time = time.time()

    # wait for new image
    (frame_time, frame) = captureQueue.get(block=True, timeout=None)

    # display and transmit
    if (current_time - last_rtp) >= rtp_interval:
        # annotate image
        frame_rtp=frame.copy()
        cv2.putText(frame_rtp,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation, font, fontScale, fontColor, lineType)
        # transmit image
        if not rtpQueue.full():
            rtpQueue.put((frame_time, frame_rtp), block=False) 
        else:
            logger.log(logging.WARNING, "Status:rtp Queue is full!")

        # show the captured images
        cv2.imshow(window_name, frame_rtp)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        last_rtp = current_time

camera.stop()
rtp.stop()
cv2.destroyAllWindows()
