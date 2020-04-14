import cv2
import logging
import time
import platform
import numpy as np
from   screeninfo import get_monitors

import time

logging.basicConfig(level=logging.DEBUG)

monitor = get_monitors()
while monitor==[]:
    monitor = get_monitors()

display_width   = int(monitor[0].width * 0.8)
display_height  = int(monitor[0].height * 0.8)

# reate camera interface
cap1 = cv2.VideoCapture('videotestsrc ! videoconvert ! appsink', apiPreference=cv2.CAP_GSTREAMER)
cap2 = cv2.VideoCapture('videotestsrc ! videoconvert ! appsink', apiPreference=cv2.CAP_GSTREAMER)

print("Getting Frames")
num_frames = 0
window_handle = cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
last_fps_time = time.time()


while(cv2.getWindowProperty("Camera", 0) >= 0):
    current_time = time.time()
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    if  frame1 is None or frame2 is None:
        pass
    else:
        num_frames += 1
        # fx=display_width/(frame1.shape[1] + frame2.shape[1])
        fx=2
        tmp1=cv2.resize(frame1, None, fx=fx, fy=fx )
        tmp2=cv2.resize(frame2, None, fx=fx, fy=fx )
        display_frame=np.concatenate((frame1,frame2), axis=1)
        cv2.putText(display_frame, "Bucket View 2020", ( 10, display_frame.shape[0] - 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1) 
        cv2.imshow('Camera', display_frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
        measured_fps = num_frames/5.0
        print( "Status:FPS:{}".format(measured_fps))
        num_frames = 0
        last_fps_time = current_time
cap1.release()
cap2.release()
cv2.destroyAllWindows()

