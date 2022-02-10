###############################################################################
# CSI video capture on Jetson Nano using gstreamer
# Latency can be 100-200ms
# Urs Utzinger, 
#
# 2022, First release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

# System
import logging, time, os, subprocess

# Open Computer Vision
import cv2

# platform system
import platform

###############################################################################
# Video Capture
###############################################################################

class libcameraCapture(Thread):
    """
    This thread continually captures frames from a CSI camera on Jetson Nano
    """

    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    def __init__(self, configs, 
        camera_num: int = 0, 
        res: tuple = None,    # width, height
        exposure: float = None,
        queue_size: int = 32):

        # populate desired settings from configuration file or function call
        ####################################################################
        self._camera_num = camera_num
        if exposure is not None:
            self._exposure   = exposure  
        else: 
            self._exposure   = configs['exposure']
        if res is not None:
            self._camera_res = res
        else: 
            self._camera_res = configs['camera_res']
        self._capture_width  = self._camera_res[0] 
        self._capture_height = self._camera_res[1]
        self._output_res     = configs['output_res']
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._framerate      = configs['fps']
        self._flip_method    = configs['flip']

        # Threading Queue
        self.capture         = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True
        self.cam_lock        = Lock()

        # open up the camera
        self._open_cam()

        # Init vars
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        Thread.__init__(self)

    # Thread routines 
    # Start Stop and Update Thread
    ###################################################################
    def stop(self): 
        """stop the thread"""
        self.stopped = True

    def start(self, capture_queue = None):
        """set the thread start conditions"""
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_frames = 0
        while not self.stopped:
            current_time = time.time()

            # Get New Image
            if self.cam is not None:
                with self.cam_lock:
                    _, img = self.cam.read()
                num_frames += 1
                self.frame_time = int(current_time*1000)
            
                if (img is not None) and (not self.capture.full()):
                    self.capture.put_nowait((current_time*1000., img))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.WARNING, "libcameraCap:Capture Queue is full!"))

            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "NanoCap:FPS:{}".format(self.measured_fps)))
                last_time = current_time
                num_frames = 0

        self.cam.release()

    def gstreamer_pipeline(self,
            camera_num:     int   = 0,
            capture_width:  int   = 1920, 
            capture_height: int   = 1080,
            output_width:   int   = 1280, 
            output_height:  int   = 720,
            framerate:      float = 30., 
            exposure_time:  float = -1,
            flip_method:    int   = 0):

            """
            Create gstreamer pipeline string
            """
            ###################################################################################
            # gstreamer Examples
            ###################################################################################

            # libcamerasrc 
            # ! video/x-raw, width=, height=, framerate=(fraction)0/1
            # ! videoconvert 
            # ! videoscale 
            # ! video/x-raw, width=(int) , height=(int)" 
            # ! appsink

            # libcamerasrc ! 
            # capsfilter caps=video/x-raw,width=1280,height=720,format=NV12 
            # ! v4l2convert 
            # ! v4l2h264enc extra-controls="controls,repeat_sequence_header=1" 
            # ! h264parse 
            # ! rtph264pay 
            # ! udpsink host=localhost port=5000

            # gst-launch-1.0 libcamerasrc num-buffers=-1 
            # ! video/x-raw,width=640,height=480, framerate=30/1 
            # ! videoconvert 
            # ! jpegenc 
            # ! tcpserversink  host=192.168.178.32 port=5000

            if framerate == 120:
                capture_width = 1280
                capture_height = 720
                sensormode = 5
            else:
                sensormode = -1

            if exposure_time <= 0:
                pass

            # deal with auto resizing
            if output_height <= 0: output_height = capture_height
            if output_width  <=0:  output_width  = capture_width

            ###################################################################################
            # libcamerasrc
            # https://github.com/kbingham/libcamera
            ###################################################################################

            libcamerasrc_str = (
                'libcamerasrc '                                     +
                'camera-name="Camera {:d}" '.format(camera_num)     + 
                'name="libcamerasrc{:d}" '.format(camera_num)       +
            )

            ###################################################################################
            # vide0/x-raw
            # https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html
            # videoscale
            #  width, height
            # videoflip
            #  method = 
            #   none (0) – Identity (no rotation)
            #   clockwise (1) – Rotate clockwise 90 degrees
            #   rotate-180 (2) – Rotate 180 degrees
            #   counterclockwise (3) – Rotate counter-clockwise 90 degrees
            #   horizontal-flip (4) – Flip horizontally
            #   vertical-flip (5) – Flip vertically
            #   upper-left-diagonal (6) – Flip across upper left/lower right diagonal
            #   upper-right-diagonal (7) – Flip across upper right/lower left diagonal
            #   automatic (8) – Select flip method based on image-orientation tag
            ###################################################################################

            if   self._flip_method == 0: # no flipping
                flipstr = '! videoflip method=0 '
            elif self._flip_method == 1: # ccw 90
                flipstr = '! videoflip method=3 '
            elif self._flip_method == 2: # rot 180, same as flip lr & up
                flipstr = '! videoflip method=2 '
            elif self._flip_method == 3: # cw 90
                flipstr = '! videoflip method=1 '
            elif self._flip_method == 4: # horizontal
                flipstr = '! videoflip method=4 '
            elif self._flip_method == 5: # upright diagonal. ccw & lr
                flipstr = '! videoflip method=7 '
            elif self._flip_method == 6: # vertical
                flipstr = '! videoflip method=5 '
            elif self._flip_method == 7: # upperleft diagonal
                flipstr = '! videoflip method=6 '
            
            gstreamer_str = (
                '! video/x-raw, '                                                                       +
                'width=(int){:d}, ',format(capture_width)                                               +
                'height=(int){:d}, '.format(capture_height)                                             +
                'framerate=(fraction){:d}/1, '.format(framerate)                                        +
                # 'max-framerate=(fraction){:d}/1, '.format(something)                                    +
                'views=1, '                                                                             +
                # 'interlatce-mode="progrfessive", '                                                      +
                # 'chroma-site="", '                                                                      +
                # 'colorimetry="", '                                                                      +
                'pixel-aspect-ratio=1/1, '                                                              +
                'format={%s} '.format('NV12')                                                           +
                '! videoconvert '                                                                       +
                '! videoscale '                                                                         +
                '! video/x-raw, width=(int){:d}, height=(int){:d} '.format(output_width, output_height) +
                flipstr +
                '! appsink')

            return ( libcamerasrc_str + gstreamer_str )

    #
    # Setup the Camera
    ############################################################################
    def _open_cam(self):
        """
        Open up the camera so we can begin capturing frames
        """
        self.gst=self.gstreamer_pipeline(
            camera_num     = self._camera_num,
            capture_width  = self._capture_width,
            capture_height = self._capture_height,
            output_width   = self._output_width,
            output_height  = self._output_height,
            framerate      = self._framerate,
            exposure_time  = self._exposure,
            flip_method    = self._flip_method)

        if not self.log.full(): self.log.put_nowait((logging.INFO, self.gst))

        self.cam = cv2.VideoCapture(self.gst, cv2.CAP_GSTREAMER)

        self.cam_open = self.cam.isOpened()

        if not self.cam_open:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "NanoCap:Failed to open camera!"))

    # Camera Routines
    ##################################################################

    # None yet

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    configs = {
        'camera_res'      : (1280, 720),    # width & height
        'exposure'        : -1,             # microseconds, internally converted to nano seconds, <= 0 autoexposure
        'fps'             : 60,             # can not get more than 60fps
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 6,              # 0=norotation 
                                            # 1=ccw90deg 
                                            # 2=rotation180 
                                            # 3=cw90 
                                            # 4=horizontal 
                                            # 5=upright diagonal flip 
                                            # 6=vertical 
                                            # 7=uperleft diagonal flip
        'displayfps'       : 30
    }

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Raspi libcamera Capture")

    logger.log(logging.DEBUG, "Starting Capture")

    plat = platform.system() 
    if plat == 'Linux':
        platform.machine() == "armv7l":
    
    camera = libcameraCapture(configs, camera_num=0)
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")
    
    window_name = "Raspi libcamera CSI Camera"
    window_handle = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    last_display = time.perf_counter()
    stop = False    
    while not stop:
        current_time = time.perf_counter()

        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if (current_time - last_display) >= display_interval:
            cv2.imshow('Camera', frame)
            last_display = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):  stop = True
        
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0: stop = True
        except: stop = True
         
        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "{}".format(msg))

    camera.stop()
    cv2.destroyAllWindows()