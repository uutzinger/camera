 
###############################################################################
# RTSP video server
# BitBuckets FRC 4183
# 2019
###############################################################################

Work in progress

gst-inspect-1.0 omxh264enc
pipeline_out = "appsrc ! videoconvert ! video/x-raw, framerate=20/1, format=RGBA ! glimagesink sync=false"
fourcc = cv2.VideoWriter_fourcc(*'H264')
stream_out = cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, 0, 20.0, (1280,720))
while True:
    ret, frame = stream_in.read()
    if ret:
      stream_out.write(frame)
      cv2.waitKey(1)

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

# Open Computer Vision
import cv2


###############################################################################
# Video Server
###############################################################################

class rtspServer(Thread):
    """
    """

    # Initialize the Camera Thread
    # Opens Capture Device
    def __init__(self, port: (int) = 554):
        # initialize 
        self.logger     = logging.getLogger("cv2Capture{}".format(camera_num))

        # populate desired settings from configuration file or function call
        if rtsp is not None: self._rtsp = RTSP
        else:                self._rtsp = configs['rtsp']
        self._display_res                         = configs['output_res']
        self._display_width                       = self._display_res[0]
        self._display_height                      = self._display_res[1]
        self._flip_method                         = configs['flip']

        # Threading Locks, Events
        self.capture_lock    = Lock() # before changing capture settings lock them
        self.frame_lock      = Lock() # When copying frame, lock it
        self.stopped         = True

        # open up the server
        self._open_server()

        # Init Frame and Thread
        self.frame     = None
        self.new_frame = False
        self.stopped   = False
        self.measured_fps = 0.0

        Thread.__init__(self)

    def _open_server(self):
        """Open up the camera so we can begin capturing frames"""

        # Open the camera with platform optimal settings
        if sys.platform.startswith('win'):
        elif sys.platform.startswith('darwin'):
        elif sys.platform.startswith('linux'):
        else:


        gst = "rtspsrc location=" + rtsp + "latency=200 ! queue ! rtph264depay ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
        self.capture = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)
        self.capture_open = self.capture.isOpened()

        # self.cv2SettingsDebug() # check camera properties

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
# Testing ##################################################
#
###################################################################
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print("Starting Capture")
    camera = rtspServer(port=554)     
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
