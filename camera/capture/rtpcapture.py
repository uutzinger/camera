###############################################################################
# RTP point to point video capture
# Uses opencv video capture to capture rtp stream
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
# 2021
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock

#
import logging
import time
import platform

# Open Computer Vision
import cv2

###############################################################################
# RTP Capture
###############################################################################

class rtpCapture(Thread):
    """
    This thread continually captures frames from a point to point RTP stream
    This is not capturing rtsp.
    """

    # Initialize the Camera Thread
    # Opens Capture Device
    def __init__(self, port: int = 554, gpu: (bool) = False):

        # initialize 
        self.logger     = logging.getLogger("rtpCapture")

        # populate settings
        self._port           = port
        self._gpuavail       = gpu

        # Threading Locks, Events
        self.capture_lock    = Lock() # before changing capture settings lock them
        self.frame_lock      = Lock() # When copying frame, lock it
        self.stopped         = True

        # open up the stream
        self._open_capture()

        # Init Frame and Thread
        self.frame        = None
        self.new_frame    = False
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        Thread.__init__(self)

    def _open_capture(self):
        """
        Open up the camera so we can begin capturing frames
        """

        ## Open the camera with platform optimal settings

        # https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/

        gst = 'udpsrc port={:d} caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtph264depay ! '.format(self._port)
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                gst = gst + 'h264parse ! omxh264dec ! nvvidconv ! '
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = gst + 'h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        elif:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'decodebin ! videoconvert ! '
        gst = gst + 'appsink sync=false'
        self.logger.log(logging.INFO, gst)
        self.capture = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        self.capture_open = self.capture.isOpened()        
        if not self.capture_open:
            self.logger.log(logging.CRITICAL, "Status:Failed to open rtp stream!")

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self, capture_queue = None):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=(capture_queue, ))
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self, capture_queue):
        """ run the thread """
        last_fps_time = time.time()
        num_frames = 0
        while not self.stopped:
            current_time = time.time()

            with self.capture_lock:
                _, img = self.capture.read()
                num_frames += 1
                self.frame_time = int(current_time*1000)

            if img is not None:
                if capture_queue is not None:
                    if not capture_queue.full():
                        capture_queue.put((self.frame_time, img), block=False)
                    else:
                        self.logger.log(logging.WARNING, "Status:Capture Queue is full!")                                    
                else:
                    self.frame = img

            # FPS calculation
            if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.logger.log(logging.INFO, "Status:FPS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time

            if self.stopped:
                self.capture.release()        

    #
    # Frame routines
    # If queue is not used
    #########################################################################

    @property
    def frame(self):
        """ returns most recent frame """
        with self.frame_lock:
            self._new_frame = False
        return self._frame
    @frame.setter
    def frame(self, val):
        """ set new frame content """
        with self.frame_lock:
            self._frame = val
            self._new_frame = True

    @property
    def new_frame(self):
        """ check if new frame available """
        with self.frame_lock:
            return self._new_frame
    @new_frame.setter
    def new_frame(self, val):
        """ override wether new frame is available """
        with self.frame_lock:
            self._new_frame = val
