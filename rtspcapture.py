 
###############################################################################
# RTSP video capture
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuation of codec
# BitBuckets FRC 4183
# 2019
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock
from threading import Event

#
import logging
import time
import sys
import platform

# Open Computer Vision
import cv2

# Camera configuration file
from configs   import configs

###############################################################################
# Video Capture
###############################################################################

class rtspCapture(Thread):
    """
    This thread continually captures frames from an RTSP stream
    """

    # Initialize the Camera Thread
    # Opens Capture Device
    def __init__(self, rtsp: (str) = None):
        # initialize 
        self.logger     = logging.getLogger("rtspCapture")

        # populate desired settings from configuration file or function call
        if rtsp is not None: self._rtsp = rtsp
        else:                self._rtsp = configs['rtsp']
        self._display_res               = configs['output_res']
        self._display_width             = self._display_res[0]
        self._display_height            = self._display_res[1]
        self._flip_method               = configs['flip']

        # Threading Locks, Events
        self.capture_lock    = Lock() # before changing capture settings lock them
        self.frame_lock      = Lock() # When copying frame, lock it
        self.stopped         = True

        # open up the camera
        self._open_capture()

        # Init Frame and Thread
        self.frame     = None
        self.new_frame = False
        self.stopped   = False
        self.measured_fps = 0.0

        Thread.__init__(self)

    def _open_capture(self):
        """
        Open up the camera so we can begin capturing 
        For testing:
            Anywhere:
                gst-launch-1.0 rtspsrc location=rtsp://localhost:1181/camera ! fakesink
            Windows:
                Install gstreamer from https://gstreamer.freedesktop.org/data/pkg/windows/
                and select 	gstreamer-1.0-msvc-x86_64-_latestversion_.msi
                Install Custom to C:/gstreamer
                add C:\gstreamer\1.0\x86_64\bin to Path variable in Environment Variables
                gst-launch-1.0 playbin uri=rtsp://localhost:1181/camera
                gst-launch-1.0 rtspsrc location=rtsp://192.168.11.26:1181/camera latency=400 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
                !!! print(cv2.getBuildInformation()) !!! gstreamer needs to be on
                If not follow https://medium.com/@galaktyk01/how-to-build-opencv-with-gstreamer-b11668fa09c
            Raspian: 
                Make sure regular gstramer in Buster is installed
                gst-launch-1.0 rtspsrc location=rtsp://localhost:1181/camera latency=100 ! rtph264depay ! h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! autovideosink sync=false
            Example with authentication:
                gst-launch-1.0 rtspsrc location=rtsp://user:pass@192.168.81.32:554/live/ch00_0 ! rtph264depay ! h264parse ! decodebin ! autovideosink
            JetsonNano:
                gst-launch-1.0 rtspsrc location= rtsp://192.168.8.50:8554/unicast latency=1000 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! autovideosink sync=false
        """

        plat = platform.system()
        if plat == "Windows":
            # gst = 'rtspsrc location=' + self._rtsp + ' latency=400 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false'
            # self.capture = cv2.VideoCapture(gst)
            self.capture = cv2.VideoCapture(self._rtsp) # uses ffmpeg subsystem, but windows subsystem does not handle circular bufers
        elif plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                gst ='rtspsrc location=' + self._rtsp + ' latency=10 ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false'
                self.capture = cv2.VideoCapture(gst)
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = 'rtspsrc location=' + self._rtsp + ' latency=400 ! queue ! rtph264depay ! h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! appsink sync=false'
                # might not need the two queue statements above
                self.capture = cv2.VideoCapture(gst)
        elif plat == "MacOS":
            gst = 'rtspsrc location=' + self._rtsp + ' latency=400 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink'
            self.capture = cv2.VideoCapture(gst)
        else:
            gst = 'rtspsrc location=' + self._rtsp + ' latency=400 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink'
            self.capture = cv2.VideoCapture(gst)

        self.capture_open = self.capture.isOpened()        
        if not self.capture_open:
            self.logger.log(logging.CRITICAL, "Status:Failed to open camera!")

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=())
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        last_fps_time = time.time()
        last_exposure_time = last_fps_time
        num_frames = 0
        while not self.stopped:
            current_time = time.time()
            # FPS calculation
            if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.logger.log(logging.DEBUG, "Status:FPS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time
            with self.capture_lock:
                _, img = self.capture.read()

            if img is not None:
                # adjust output height
                if self._display_height > 0:
                    # tmp = cv2.resize(img, self._display_res, interpolation = cv2.INTER_NEAREST)
                    tmp = cv2.resize(img, self._display_res)
                else:
                    tmp = img
                # flip image if needed
                if   self._flip_method == 1: # ccw 90
                    self.frame = cv2.roate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 2: # rot 180, same as flip lr & up
                    self.frame = cv2.roate(tmp, cv2.ROTATE_180)
                elif self._flip_method == 3: # cw 90
                    self.frame = cv2.roate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 4: # horizontal
                    self.frame = cv2.flip(tmp, 0)
                elif self._flip_method == 5: # upright diagonal. ccw & lr
                    tmp = cv2.roate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 6: # vertical
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 7: # upperleft diagonal
                    self.frame = cv2.transpose(tmp)
                else:
                    self.frame = tmp

                num_frames += 1

            if self.stopped:
                self.capture.release()        

    #
    # Frame routines ##################################################
    # Each camera stores frames locally
    ###################################################################

    @property
    def frame(self):
        """ returns most recent frame """
        self._new_frame = False
        return self._frame

    @frame.setter
    def frame(self, img):
        """ set new frame content """
        with self.frame_lock:
            self._frame = img
            self._new_frame = True

    @property
    def new_frame(self):
        """ check if new frame available """
        out = self._new_frame
        return out

    @new_frame.setter
    def new_frame(self, val):
        """ override wether new frame is available """
        self._new_frame = val

#
# Testing ########################################################
###################################################################
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print("Starting Capture")

    camera = rtspCapture(rtsp='rtsp://192.168.11.26:1181/camera')     
    camera.start()

    print("Getting Frames")
    window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty("Camera", 0) >= 0):
        if camera.new_frame:
            cv2.imshow('Camera', camera.frame)
            # print(camera.frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()
