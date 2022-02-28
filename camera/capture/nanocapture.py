###############################################################################
# CSI video capture on Jetson Nano using gstreamer
# Latency can be 100-200ms
# Urs Utzinger, 
#
# 2021, Initialize
# 2020, Update Queue
# 2019, First release
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

###############################################################################
# Video Capture
###############################################################################

class nanoCapture(Thread):
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
                    #flip and resize is done in gstreamer
                    self.capture.put_nowait((current_time*1000., img))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.WARNING, "NanoCap:Capture Queue is full!"))

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
            exposure_time:  float = -1,    # microseconds
            flip_method:    int   = 0):

            """
            Create gstreamer pipeline string
            """
            ###################################################################################
            # gstreamer Options 
            # Examples for IX279
            ###################################################################################

            ###################################################################################
            # nvarguscamerasrc
            ###################################################################################
            # name                : The name of the object
            #                     flags: readable, writable
            #                     String. Default: "nvarguscamerasrc0"
            # parent              : The parent of the object
            #                     flags: readable, writable
            #                     Object of type "GstObject"
            # *blocksize           : Size in bytes to read per buffer (-1 = default)
            #                     flags: readable, writable
            #                     Unsigned Integer. Range: 0 - 4294967295 Default: 4096 
            #        'blocksize=-1 ' 
            # *num-buffers         : Number of buffers to output before sending EOS (-1 = unlimited)
            #                     flags: readable, writable
            #                     Integer. Range: -1 - 2147483647 Default: -1 
            #        'num-buffers=-1 '
            # typefind            : Run typefind before negotiating (deprecated, non-functional)
            #                     flags: readable, writable, deprecated
            #                     Boolean. Default: false
            # do-timestamp        : Apply current stream time to buffers
            #                     flags: readable, writable
            #                     Boolean. Default: true
            # *silent              : Produce verbose output ?
            #                     flags: readable, writable
            #                     Boolean. Default: true
            # *timeout             : timeout to capture in seconds (Either specify timeout or num-buffers, not both)
            #                     flags: readable, writable
            #                     Unsigned Integer. Range: 0 - 2147483647 Default: 0 
            #        'timeout=0 ' 
            # *wbmode              : White balance affects the color temperature of the photo
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamWBMode" Default: 1, "auto"
            #                         (0): off              - GST_NVCAM_WB_MODE_OFF
            #                         (1): auto             - GST_NVCAM_WB_MODE_AUTO
            #                         (2): incandescent     - GST_NVCAM_WB_MODE_INCANDESCENT
            #                         (3): fluorescent      - GST_NVCAM_WB_MODE_FLUORESCENT
            #                         (4): warm-fluorescent - GST_NVCAM_WB_MODE_WARM_FLUORESCENT
            #                         (5): daylight         - GST_NVCAM_WB_MODE_DAYLIGHT
            #                         (6): cloudy-daylight  - GST_NVCAM_WB_MODE_CLOUDY_DAYLIGHT
            #                         (7): twilight         - GST_NVCAM_WB_MODE_TWILIGHT
            #                         (8): shade            - GST_NVCAM_WB_MODE_SHADE
            #                         (9): manual           - GST_NVCAM_WB_MODE_MANUAL
            # *saturation          : Property to adjust saturation value
            #                     flags: readable, writable
            #                     Float. Range:               0 -               2 Default:               1 
            #        'saturation=1 ' 
            # sensor-id           : Set the id of camera sensor to use. Default 0.
            #                     flags: readable, writable
            #                     Integer. Range: 0 - 255 Default: 0 
            # *sensor-mode         : Set the camera sensor mode to use. Default -1 (Select the best match)
            #                     flags: readable, writable
            #                     Integer. Range: -1 - 255 Default: -1 
            #                     # -1..255, IX279 
            #                     # 0 (3264x2464,21fps)
            #                     # 1 (3264x1848,28fps)
            #                     # 2 (1080p, 30fps)
            #                     # 3 (1640x1232 30fps)
            #                     # 4 (720p, 60fps)
            #                     # 5 (720p, 120fps)
            # total-sensor-modes  : Query the number of sensor modes available. Default 0
            #                     flags: readable
            #                     Integer. Range: 0 - 255 Default: 0 
            # exposuretimerange   : Property to adjust exposure time range in nanoseconds
            #         Use string with values of Exposure Time Range (low, high)
            #         in that order, to set the property.
            #         eg: exposuretimerange="34000 358733000"
            #                     flags: readable, writable
            #                     String. Default: null
            # *gainrange           : Property to adjust gain range
            #         Use string with values of Gain Time Range (low, high)
            #         in that order, to set the property.
            #         eg: gainrange="1 16"
            #                     flags: readable, writable
            #                     String. Default: null
            #        'gainrange="1.0 10.625" '
            # * ispdigitalgainrange : Property to adjust digital gain range
            #         Use string with values of ISP Digital Gain Range (low, high)
            #         in that order, to set the property.
            #         eg: ispdigitalgainrange="1 8"
            #                     flags: readable, writable
            #                     String. Default: null
            #        'ispdigitalgainrange="1 8" '               
            # *tnr-strength        : property to adjust temporal noise reduction strength
            #                     flags: readable, writable
            #                     Float. Range:              -1 -               1 Default:              -1 
            #        'tnr-strength=-1 '
            # *tnr-mode            : property to select temporal noise reduction mode
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamTNRMode" Default: 1, "NoiseReduction_Fast"
            #                         (0): NoiseReduction_Off - GST_NVCAM_NR_OFF
            #                         (1): NoiseReduction_Fast - GST_NVCAM_NR_FAST
            #                         (2): NoiseReduction_HighQuality - GST_NVCAM_NR_HIGHQUALITY
            #        'tnr-mode=1 '
            # *ee-mode             : property to select edge enhnacement mode
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamEEMode" Default: 1, "EdgeEnhancement_Fast"
            #                         (0): EdgeEnhancement_Off - GST_NVCAM_EE_OFF
            #                         (1): EdgeEnhancement_Fast - GST_NVCAM_EE_FAST
            #                         (2): EdgeEnhancement_HighQuality - GST_NVCAM_EE_HIGHQUALITY
            #        #'ee-mode=0' 
            # *ee-strength         : property to adjust edge enhancement strength
            #                     flags: readable, writable
            #                     Float. Range:              -1 -               1 Default:              -1 
            #        #'ee-strength=-1 '
            # *aeantibanding       : property to set the auto exposure antibanding mode
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamAeAntiBandingMode" Default: 1, "AeAntibandingMode_Auto"
            #                         (0): AeAntibandingMode_Off - GST_NVCAM_AEANTIBANDING_OFF
            #                         (1): AeAntibandingMode_Auto - GST_NVCAM_AEANTIBANDING_AUTO
            #                         (2): AeAntibandingMode_50HZ - GST_NVCAM_AEANTIBANDING_50HZ
            #                         (3): AeAntibandingMode_60HZ - GST_NVCAM_AEANTIBANDING_60HZ
            #        'aeantibanding=1 '
            # *exposurecompensation: property to adjust exposure compensation
            #                     flags: readable, writable
            #                     Float. Range:              -2 -               2 Default:               0 
            #        'exposurecompensation=0 '                  # -2..2
            # *aelock              : set or unset the auto exposure lock
            #                     flags: readable, writable
            #                     Boolean. Default: false
            #        'aelock=true '
            # *awblock             : set or unset the auto white balance lock
            #                     flags: readable, writable
            #                     Boolean. Default: false
            #        'awblock=false ' 
            # *bufapi-version      : set to use new Buffer API
            #                     flags: readable, writable
            #                     Boolean. Default: false
            #        'bufapi-version=false '
            ###################################################################################

            # if framerate > 60:
            #     sensormode = 5
            # else:
            #     sensormode = -1

            if exposure_time <= 0:
                # auto exposure
                ################
                nvarguscamerasrc_str = (
                    'nvarguscamerasrc '                             +
                    'sensor-id={:d} '.format(camera_num)            +
                    'name="NanoCam{:d}" '.format(camera_num)        +
                    'do-timestamp=true '                            +
                    # 'sensor-mode={:d} '.format(sensormode)          +
                    'timeout=0 '                                    +
                    'blocksize=-1 '                                 +
                    'num-buffers=-1 '                               +
                    'tnr-strength=-1 '                              +
                    'tnr-mode=1 '                                   +
                    'aeantibanding=1 '                              +
                    'bufapi-version=false '                         +
                    'silent=true '                                  +
                    'saturation=1 '                                 +
                    'wbmode=1 '                                     +
                    'awblock=false '                                +
                    'aelock=false '                                 +
                    'ee-mode=0 '                                    +
                    'ee-strength=-1 '                               +
                    'exposurecompensation=0 '                       +
                    'exposuretimerange="34000 358733000" '
                )
            else:
                # static exposure
                #################
                nvarguscamerasrc_str = (  
                    'nvarguscamerasrc '                             +
                    'sensor-id={:d} '.format(camera_num)            +
                    'name="NanoCam{:d}" '.format(camera_num)        +
                    'do-timestamp=true '                            +
                    'timeout=0 '                                    +
                    'blocksize=-1 '                                 +
                    'num-buffers=-1 '                               +
                    # 'sensor-mode={:d} '.format(sensormode)          +
                    'tnr-strength=-1 '                              +
                    'tnr-mode=1 '                                   +
                    'aeantibanding=1 '                              +
                    'bufapi-version=false '                         +
                    'silent=true '                                  +
                    'saturation=1 '                                 +
                    'wbmode=1 '                                     +
                    'awblock=false '                                +
                    'aelock=true '                                  +
                    'ee-mode=0 '                                    +
                    'ee-strength=-1 '                               +
                    'exposurecompensation=0 '                       +
                    'exposuretimerange="{:d} {:d}" '.format(exposure_time*1000,exposure_time*1000) 
                )

            # flip-method         : video flip methods
            #                         flags: readable, writable, controllable
            #                         Enum "GstNvVideoFlipMethod" Default: 0, "none"
            #                         (0): none             - Identity (no rotation)
            #                         (1): counterclockwise - Rotate counter-clockwise 90 degrees
            #                         (2): rotate-180       - Rotate 180 degrees
            #                         (3): clockwise        - Rotate clockwise 90 degrees
            #                         (4): horizontal-flip  - Flip horizontally
            #                         (5): upper-right-diagonal - Flip across upper right/lower left diagonal
            #                         (6): vertical-flip    - Flip vertically
            #                         (7): upper-left-diagonal - Flip across upper left/lower right 

            # deal with auto resizing
            if output_height <= 0:
                output_height = capture_height
            if output_width <=0:
                output_width = capture_width

            gstreamer_str = (
                '! video/x-raw(memory:NVMM), '                       +
                'width=(int){:d}, '.format(capture_width)            +
                'height=(int){:d}, '.format(capture_height)          +
                'format=(string)NV12, '                              +
                'framerate=(fraction){:d}/1 '.format(framerate)      +
                '! nvvidconv flip-method={:d} '.format(flip_method)  +
                '! video/x-raw, width=(int){:d}, height=(int){:d}, format=(string)BGRx '.format(output_width,output_height) +
                '! videoconvert '                                    +
                '! video/x-raw, format=(string)BGR '                 +
                '! appsink')

            return (
                nvarguscamerasrc_str + gstreamer_str
            )

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

    # OpenCV interface
    # Works for Sony IX219
    #cap.get(cv2.CAP_PROP_BRIGHTNESS)
    #cap.get(cv2.CAP_PROP_CONTRAST)
    #cap.get(cv2.CAP_PROP_SATURATION)
    #cap.get(cv2.CAP_PROP_HUE)
    #cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #cap.get(cv2.CAP_PROP_FPS)

    #V4L2 interface
    #Works for Sonty IX219
    #v4l2-ctl --set-ctrl exposure= 13..683709
    #v4l2-ctl --set-ctrl gain= 16..170
    #v4l2-ctl --set-ctrl frame_rate= 2000000..120000000
    #v4l2-ctl --set-ctrl low_latency_mode=True
    #v4l2-ctl --set-ctrl bypass_mode=Ture
    #os.system("v4l2-ctl -c exposure_absolute={} -d {}".format(val,self._camera_num))

    # Read properties

    @property
    def exposure(self):              
        if self.cam_open:
            # return os.system("v4l2-ctl -C exposure_absolute -d {}".format(self._camera_num))
            res = subprocess.run(["v4l2-ctl", "-C exposure_absolute -d {}".format(self._camera_num)], stdout=subprocess.PIPE).stdout.decode('utf-8')
            return res
        else: return float("NaN")
    @exposure.setter
    def exposure(self, val):
        if val is None:
            return
        val = int(val)
        if self.cam_open:
            with self.cam_lock:
                res = subprocess.run(["v4l2-ctl -c exposure_absolute={} -d {}".format(val, self._camera_num)], stdout=subprocess.PIPE).stdout.decode('utf-8')
                # os.system("v4l2-ctl -c exposure_absolute={} -d {}".format(val, self._camera_num))
                self._exposure = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "NanoCap:Exposure:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.ERROR, "NanoCap:Failed to set exposure to{}!".format(val)))

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

    if configs['displayfps'] >= configs['fps']:  display_interval = 0
    else:                                        display_interval = 1.0/configs['displayfps']
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Nano Capture")

    logger.log(logging.DEBUG, "Starting Capture")

    camera = nanoCapture(configs, camera_num=0)
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")

    window_handle = cv2.namedWindow("Nano CSI Camera", cv2.WINDOW_AUTOSIZE)

    last_display = time.perf_counter()

    stop = False  
    while(not stop):
        current_time = time.perf_counter()

        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "NanoCap:{}".format(msg))

        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if (current_time - last_display) >= display_interval:
            cv2.imshow('Nano CSI Camera', frame)
            last_display = current_time
            if cv2.waitKey(1) & 0xFF == ord('q'):  stop=True
            #try: 
            #    if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0: 
            #        stop = True
            #except: 
            #    stop = True

    camera.stop()
    cv2.destroyAllWindows()