import cv2
import logging
import time
import platform

loop_interval = 1.0/200.0

# Dell Inspiron 15 internal camer
# from configs.dell_internal_configs  import configs as configs
# Eluktronics Max-15 internal camera
from configs.eluk_configs import configs as configs
# Nano Jetson IMX219 camera
# from configs.nano_IMX219_configs  import configs as configs
# Raspberry Pi v1 & v2 camera
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs
# ELP MAX15 internal camera
# from configs.ELP1080p_configs  import configs as configs
 
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

# Create camera interface based on computer OS you are running
plat = platform.system()
if plat == 'Windows': 
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)
elif plat == 'Linux':
    if platform.machine() == "aarch64": # for me this is jetson nano
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is raspberry for me
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs)
elif plat == 'MacOS':
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)
else:
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs)

print("Getting Images")
camera.start()

# Initialize Variables
last_display   = time.time()
last_fps_time  = time.time()
measured_dps   = 0
num_frames_received    = 0
num_frames_displayed   = 0

while(cv2.getWindowProperty(window_name, 0) >= 0):
    current_time = time.time()
    start_time   = time.perf_counter()

    if camera.new_frame:
        frame = camera.frame
        frame_time = camera.frame_time
        num_frames_received += 1

        if (current_time - last_display) >= display_interval:
            cv2.putText(frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
            cv2.putText(frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
            cv2.imshow(window_name, frame)
            # quit the program if users enter q or closes the display window
            # the waitKey function limits the display frame rate to about 30fps for me
            # without waitKey the opencv window is not refreshed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_display = current_time
            num_frames_displayed += 1

    if current_time - last_fps_time >= dps_measure_time:
        measured_fps = num_frames_received/dps_measure_time
        logger.log(logging.INFO, "Status:Frames received per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/dps_measure_time
        logger.log(logging.INFO, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_fps_time = current_time

camera.stop()
cv2.destroyAllWindows()
