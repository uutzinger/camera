import cv2
import time

rtsp="rtsp://10.41.83.100:554/camera"
gst = 'rtspsrc location=' + rtsp + ' latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false'

print("Starting Capture")
# cap = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture('autovideosrc ! videoconvert ! appsink', apiPreference=cv2.CAP_GSTREAMER)

print("Getting Images")
num_frames = 0
window_handle = cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
last_fps_time = time.time()

while(cv2.getWindowProperty("Camera", 0) >= 0):
    current_time = time.time()
    ret, frame = cap.read()
    if not frame == None:
        num_frames += 1
        cv2.imshow('RTSP', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
        measured_fps = num_frames/5.0
        print( "Status:FPS:{}".format(measured_fps))
        num_frames = 0
        last_fps_time = current_time
cap.release()
cv2.destroyAllWindows()

# gst-launch-1.0 playbin uri=rtsp://10.41.83.100:554/camera
