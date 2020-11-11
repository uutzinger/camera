##########################################################################
# Testing of display fps.
##########################################################################
# Results
# =======
# Display Interval 0.01:
#   Loop time 0s
#     64.2 FPS
#     CPU Usage: 2-3%
#   Loop time 0.01s 
#     64.4 frames displayed
#     CPU Usage: 1-3%
# Display Interval 0.03:
#   Loop time 0s 
#     32.8 frames displayed
#     CPU Usage: 4%
#   Loop time 0.01s 
#     32 frames displayed
#     CPU Usage: 0.5-1.5%
##########################################################################
import logging
import time
import numpy as np
import cv2

looptime = 0.0
display_interval = 0.01
window_name = 'Camera'

test_img = np.random.randint(0, 255, (540, 720), 'uint8')
frame = np.zeros((540,720), dtype=np.uint8)

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

# Init Frame and Thread
measured_dps = 0.0

num_frames = 0 
last_dps_time = time.time()
last_display = time.time()

# Main Loop
while (cv2.getWindowProperty(window_name, 0) >= 0):
    current_time = time.time()

    if current_time - last_dps_time >= 5.0:
        measured_dps = num_frames/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        last_dps_time = current_time
        num_frames = 0

    if (current_time - last_display) > display_interval:
        frame = test_img.copy()
        cv2.putText(frame,"Frame:{}".format(num_frames),             textLocation0, font, fontScale, fontColor, lineType)
        cv2.putText(frame,"Frame Rate:{} [Hz]".format(measured_dps), textLocation1, font, fontScale, fontColor, lineType)
        cv2.imshow(window_name, frame)
        #cv2.resizeWindow(window_name, width, height)
        #cv2.moveWindow(window_name, 0, 0)
        #cv2.setWindowTitle(window_name, 'Display Server')
        num_frames += 1
        last_display = current_time

        key = cv2.waitKey(1) 
        if (key == 27) or (key & 0xFF == ord('q')):
            break

    delay_time = looptime - (time.time() - current_time) # make sure the while-loop takes at least looptime to complete
    if  delay_time >= 0.001:
        time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

# Cleanup
cv2.destroyAllWindows()
