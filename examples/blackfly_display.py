##########################################################################
# Testing of blackfly capture thread.
# The camera shuld achieve 525 frames per second.
##########################################################################
# Results
#    Capture FPS: 524.2
#    Display FPS: 32.2
#    CPU Usage: 6-7%
##########################################################################
import cv2
import logging
import time

# Camera configuration file
from configs.blackfly_configs  import configs

if configs['displayfps'] >= configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

dps_measure_time = 5.0

window_name    = 'Camera'
font           = cv2.FONT_HERSHEY_SIMPLEX
textLocation0  = (10,20)
textLocation1  = (10,60)
fontScale      = 1
fontColor      = (255,255,255)
lineType       = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Blackfly")

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
logger.log(logging.INFO, "Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
camera.start()

# Initialize Variables
last_display  = time.time()
last_time = time.time()
measured_dps  = 0
num_frames_received  = 0
num_frames_displayed = 0

stop = False
while ( (cv2.getWindowProperty(window_name, 0) >= 0) and (not stop)):
    current_time = time.time()
    # start_time   = time.perf_counter()

    # wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    num_frames_received += 1

    if current_time - last_time >= dps_measure_time:
        measured_fps = num_frames_received/dps_measure_time
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_fps))
        num_frames_received = 0
        measured_dps = num_frames_displayed/dps_measure_time
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        num_frames_displayed = 0
        last_time = current_time

    if (current_time - last_display) > display_interval:
        display_frame = frame.copy()
        cv2.putText(display_frame,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(display_frame,"Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, display_frame)
        # quit the program if users enter q or closes the display window
        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True
        last_display = current_time
        num_frames_displayed += 1

    # avoid looping unnecessarely, 
    # this is only relevant for low display frames per second
    # end_time = time.perf_counter()
    # delay_time = loop_interval - (end_time - start_time)
    # if  delay_time >= 0.005:
    #    time.sleep(delay_time)  # this creates at least 3ms delay, regardless of delay_time

camera.stop()
cv2.destroyAllWindows()
