##########################################################################
# Testing of display fps using opencv waitkey
##########################################################################
# Results
# =======
#     80-90 frames displayed
#     CPU Usage: 0.5-1.5%
##########################################################################
import logging
import time
import numpy as np
import cv2

width = 720       # 1920, 720
height = 540      # 1080, 540

display_interval = 1./100.  # lets attempt 100fps (my max is 85-90)
loop_interval = 1./200.     # update main loop every 5ms
window_name = 'Camera'

# synthetic data
test_img = np.random.randint(0, 255, (height, width), 'uint8') # random image
frame = np.zeros((height,width), dtype=np.uint8) # pre allocate

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Display")

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
font          = cv2.FONT_HERSHEY_SIMPLEX
textLocation0 = (10,20)
textLocation1 = (10,60)
fontScale     = 1
fontColor     = (255,255,255)
lineType      = 2
#cv2.resizeWindow(window_name, width, height)
#cv2.moveWindow(window_name, 0, 0)
#cv2.setWindowTitle(window_name, 'Display Server')

# Init Frame and Thread
measured_dps = 0.0          # displayed frames per second
num_frames = 0              # frame counter
dps_measure_time = 5.0      # count frames for 5 sec
last_dps_time = time.time() 
last_display = time.time()

# Main Loop
while (cv2.getWindowProperty(window_name, 0) >= 0):
    current_time = time.time()
    start_time   = time.perf_counter()

    # update displayed frames per second
    if current_time - last_dps_time >= dps_measure_time:
        measured_dps = num_frames/dps_measure_time
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        last_dps_time = current_time
        num_frames = 0

    # display frame
    if (current_time - last_display) > display_interval:
        frame = test_img.copy()
        cv2.putText(frame,"Frame:{}".format(num_frames),             textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame,"Frame Rate:{} [Hz]".format(measured_dps), textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame)
        num_frames += 1
        last_display = current_time

        key = cv2.waitKey(1) 
        if (key == 27) or (key & 0xFF == ord('q')):
            break

    # avoid looping unnecessarely, 
    # this is only relevant for low fps
    end_time = time.perf_counter()
    delay_time = loop_interval - (end_time - start_time)
    if  delay_time >= 0.005:
        time.sleep(delay_time)  # this creates at least 3ms delay, regardless of delay_time

# Cleanup
cv2.destroyAllWindows()
