 
###############################################################################
# RTSP video capture
# Uses opencv video capture to capture rtsp stream
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
# 2019
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
# RTSP Capture
###############################################################################

class rtspCapture(Thread):
    """
    This thread continually captures frames from an RTSP stream
    """

    # Initialize the Camera Thread
    # Opens Capture Device
    def __init__(self, configs, rtsp: (str) = None, gpu: (bool) = False):
        # initialize 
        self.logger     = logging.getLogger("rtspCapture")

        # populate desired settings from configuration file or function call
        if rtsp is not None: 
            self._rtsp      = rtsp
        else: 
            self._rtsp      = configs['rtsp']
        self._output_res    = configs['output_res']
        self._output_width  = self._output_res[0]
        self._output_height = self._output_res[1]
        self._flip_method   = configs['flip']
        self._gpuavail      = gpu

        # Threading Locks, Events
        self.capture_lock    = Lock() # before changing capture settings lock them
        self.frame_lock      = Lock() # When copying frame, lock it
        self.stopped         = True

        # open up the camera
        self._open_capture()

        # Init Frame and Thread
        self.frame        = None
        self.new_frame    = False
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        Thread.__init__(self)

    def _open_capture(self):
        """
        Open up the camera so we can begin capturing 
        """

        gst ='rtspsrc location=' + self._rtsp + ' latency=10 ! rtph264depay ! h264parse ! '
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                gst = gst + 'omxh264dec ! nvvidconv ! appsink sync=false'
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'avdec_h264 ! videoconvert ! '
        gst = gst + 'appsink sync=false'

        self.logger.log(logging.INFO, gst)
        self.capture = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)

        self.capture_open = self.capture.isOpened()        
        if not self.capture_open:
            self.logger.log(logging.CRITICAL, "Status:Failed to open camera!")

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
                # adjust from RGB to BGR
                # img=cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
                if (self._output_height <= 0) and (self._flip_method == 0):
                    if capture_queue is not None:
                        if not capture_queue.full():
                            capture_queue.put((self.frame_time, img), block=False)
                        else:
                            self.logger.log(logging.WARNING, "Status:Capture Queue is full!")
                    else:
                        self.frame = img
                else:
                    # adjust output height
                    tmp = cv2.resize(img, self._output_res)
                    # flip resized image
                    if   self._flip_method == 0: # no flipping
                        tmpf = tmp
                    elif self._flip_method == 1: # ccw 90
                        tmpf = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 2: # rot 180, same as flip lr & up
                        tmpf = cv2.roate(tmp, cv.ROTATE_180)
                    elif self._flip_method == 3: # cw 90
                        tmpf = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 4: # horizontal
                        tmpf = cv2.flip(tmp, 0)
                    elif self._flip_method == 5: # upright diagonal. ccw & lr
                        tmp_tmp = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                        tmpf = cv2.flip(tmp_tmp, 1)
                    elif self._flip_method == 6: # vertical
                        tmpf = cv2.flip(tmp, 1)
                    elif self._flip_method == 7: # upperleft diagonal
                        tmpf = cv2.transpose(tmp)
                    else:
                        tmpf = tmp
                    if capture_queue is not None:
                        if not capture_queue.full():
                            capture_queue.put((self.frame_time, tmpf), block=False)
                        else:
                            self.logger.log(logging.WARNING, "Status:Capture Queue is full!")                                    
                    else:
                        self.frame = tmpf

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

#
# Testing 
###################################################################
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    rtsp='rtsp://10.41.83.100:554/camera'

    print("Starting RTSP Capture")

    camera = rtspCapture(rtsp=rtsp)
       
    camera.start()

    print("Getting Frames")
    window_handle = cv2.namedWindow('RTSP', cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty('RTSP', 0) >= 0):
        if camera.new_frame:
            cv2.imshow('RTSP', camera.frame)
            # print(camera.frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()
