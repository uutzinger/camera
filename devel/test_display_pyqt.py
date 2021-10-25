##########################################################################
# Testing of display using QT5 interface.
# 

WORK IN PROGRESS

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
import PyQT5

looptime = 0.0
display_interval = 0.01
window_name = 'Camera'

test_img = np.random.randint(0, 255, (540, 720), 'uint8')
frame = np.zeros((540,720), dtype=np.uint8)

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Display")

font          = cv2.FONT_HERSHEY_SIMPLEX
textLocation0 = (10,480)
textLocation1 = (10,520)
fontScale     = 1
fontColor     = (255,255,255)
lineType      = 2

# Init Frame and Thread
measured_dps = 0.0

num_frames = 0 
last_dps_time = time.time()
last_display = time.time()

# Main Loop
while ():
    current_time = time.time()

    if current_time - last_dps_time >= 5.0:
        measured_dps = num_frames/5.0
        logger.log(logging.DEBUG, "Status:Frames displayed per second:{}".format(measured_dps))
        last_dps_time = current_time
        num_frames = 0

    if (current_time - last_display) > display_interval:
        frame = test_img.copy()
        (frame,"Frame:{}".format(num_frames),             textLocation0, font, fontScale, fontColor, lineType)
        (frame,"Frame Rate:{} [Hz]".format(measured_dps), textLocation1, font, fontScale, fontColor, lineType)
        imshow(window_name, frame)
        num_frames += 1
        last_display = current_time

    delay_time = looptime - (time.time() - current_time) # make sure the while-loop takes at least looptime to complete
    if  delay_time >= 0.001:
        time.sleep(delay_time)  # this creates at least 10-15ms delay, regardless of delay_time

# Cleanup
cv2.destroyAllWindows()


#Initialze QtGui.QImage() with arguments data, height, width, and QImage.Format
self.data = np.array(self.data).reshape(2048,2048).astype(np.int32)
qimage = QtGui.QImage(self.data, self.data.shape[0],self.data.shape[1],QtGui.QImage.Format_RGB32)
img = PrintImage(QPixmap(qimage))