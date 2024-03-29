###############################################################################
# OpenCV video capture
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
# 
# 2022 Added access to more opencv camera properties
#      Auto populates missing configs
#      Access to opencv camera configs window
# 2021 Initialize, Remove direct Frame acces (use only queue)
# 2019 Initial release, based on Bitbuckets FRC 4183 code
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

# System
import logging, time, sys

# Open Computer Vision
import cv2

###############################################################################
# Video Capture
###############################################################################

class cv2Capture(Thread):
    """
    This thread continually captures frames from a camera
    """
    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    ############################################################################
    def __init__(self, configs, 
        camera_num: int = 0, 
        res: tuple = None,    # width, height
        exposure: float = None,
        queue_size: int = 32):
        
        # populate desired settings from configuration file or function arguments
        ####################################################################
        self._camera_num       = camera_num
        if exposure is not None:
            self._exposure    = exposure  
        else: 
            if 'exposure' in configs: self._exposure       = configs['exposure']
            else:                     self._exposure       = -1.0
        if res is not None:
            self._camera_res = res
        else: 
            if 'camera_res' in configs: self._camera_res   = configs['camera_res']
            else:                       self._camera_res   = (640, 480)
        if 'output_res' in configs:     self._output_res   = configs['output_res']
        else:                           self._output_res   = (-1,-1)
        if 'fps' in configs:            self._framerate    = configs['fps']
        else:                           self._framerate    = -1.0
        if 'flip' in configs:           self._flip_method  = configs['flip']
        else:                           self._flip_method  = 0
        if 'buffersize' in configs:     self._buffersize   = configs['buffersize']         # camera drive buffer size
        else:                           self._buffersize   = -1
        if 'fourcc' in configs:         self._fourcc       = configs['fourcc']             # camera sensor encoding format
        else:                           self._fourcc       = -1
        if 'autoexposure' in configs:   self._autoexposure = configs['autoexposure']       # autoexposure depends on camera
        else:                           self._autoexposure = -1
        if 'gain' in configs:           self._gain         = configs['gain']
        else:                           self._gain         = -1.0
        if 'wb_temp' in configs:        self._wbtemp       = configs['wb_temp']
        else:                           self._wbtemp       = -1
        if 'autowb' in configs:         self._autowb       = configs['autowb']
        else:                           self._autowb       = -1
        if 'settings' in configs:       self._settings     = configs['settings']
        else:                           self._settings     = -1

        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        
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

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        # clean up

    def start(self):
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
            if self.cam is not None:
                with self.cam_lock:
                    _, img = self.cam.read()
                num_frames += 1
                self.frame_time = int(current_time*1000)

                if (img is not None) and (not self.capture.full()):

                    if (self._output_height > 0) or (self._flip_method > 0):
                        # adjust output height
                        img_resized = cv2.resize(img, self._output_res)
                        # flip resized image
                        if   self._flip_method == 0: # no flipping
                            img_proc = img_resized
                        elif self._flip_method == 1: # ccw 90
                            img_proc = cv2.roate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        elif self._flip_method == 2: # rot 180, same as flip lr & up
                            img_proc = cv2.roate(img_resized, cv2.ROTATE_180)
                        elif self._flip_method == 3: # cw 90
                            img_proc = cv2.roate(img_resized, cv2.ROTATE_90_CLOCKWISE)
                        elif self._flip_method == 4: # horizontal
                            img_proc = cv2.flip(img_resized, 0)
                        elif self._flip_method == 5: # upright diagonal. ccw & lr
                            img_proc = cv2.flip(cv2.roate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                        elif self._flip_method == 6: # vertical
                            img_proc = cv2.flip(img_resized, 1)
                        elif self._flip_method == 7: # upperleft diagonal
                            img_proc = cv2.transpose(img_resized)
                        else:
                            img_proc = img_resized # not a valid flip method
                    else:
                        img_proc = img

                    self.capture.put_nowait((self.frame_time, img_proc))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Capture Queue is full!"))


            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CAM:FPS:{}".format(self.measured_fps)))
                last_time = current_time
                num_frames = 0

        self.cam.release()

    # Setup the Camera
    ############################################################################
    def _open_cam(self):
        """
        Open up the camera so we can begin capturing frames
        """

        # Open the camera with platform optimal settings
        if sys.platform.startswith('win'):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_DSHOW) # CAP_VFW or CAP_DSHOW or CAP_MSMF or CAP_ANY
        elif sys.platform.startswith('darwin'):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('linux'):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_V4L2)
        else:
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_ANY)

        self.cam_open = self.cam.isOpened()

        if self.cam_open:
            # Apply settings to camera
            #self.height        = self._camera_res[1]   # image resolution
            #self.width         = self._camera_res[0]   # image resolution
            self.resolution    = self._camera_res      #
            self.exposure      = self._exposure        # camera exposure
            self.autoexposure  = self._autoexposure    # autoexposure
            self.fps           = self._framerate       # desired fps
            self.buffersize    = self._buffersize      # camera drive buffer size
            self.fourcc        = self._fourcc          # camera sensor encoding format
            self.gain          = self._gain            # camera gain
            self.wbtemperature = self._wbtemp          # camera white balance temperature
            self.autowb        = self._autowb          # camera enable auto white balance

            if self._settings > -1: self.cam.set(cv2.CAP_PROP_SETTINGS, 0.0) # open camera settings window
            
            # Update records
            self._camera_res    = self.resolution
            self._exposure      = self.exposure
            self._buffersize    = self.buffersize
            self._framerate     = self.fps
            self._autoexposure  = self.autoexposure
            self._fourcc        = self.fourcc
            self._fourcc_str    = self.decode_fourcc(self._fourcc)
            self._gain          = self.gain
            self._wbtemperature = self.wbtemperature
            self._autowb        = self.autowb
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to open camera!"))

    # Open Settings Window
    ############################################################################
    def opensettings(self):
        """
        Open up the camera settings window
        """
        if self.cam_open:
            self.cam.set(cv2.CAP_PROP_SETTINGS, 0.0)

    # Camera routines #################################################
    # Reading and setting camera options
    ###################################################################

    @property
    def width(self):
        """ returns video capture width """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        else: return -1
    @width.setter
    def width(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Width not changed to {}".format(val)))
            return
        if self.cam_open and val > 0:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, val):
                    # self._camera_res = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._camera_res[1]))
                    # HEIGHT and WIDTH only valid if both were set
                   if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Width to {}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Width, camera not open!"))

    @property
    def height(self):
        """ returns videocapture height """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else: return -1
    @height.setter
    def height(self, val):
        """ sets video capture height """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Height not changed:{}".format(val)))
            return
        if self.cam_open and val > 0:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val)):
                    # self._camera_res = (int(self._camera_res[0]), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    # HEIGHT and WIDTH only valid if both were set
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Height to {}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Height, camera not open!"))

    @property
    def resolution(self):
        """ returns current resolution width x height """
        if self.cam_open:
            return (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        else: return (-1, -1) 
    @resolution.setter
    def resolution(self, val):
        if val is None: return
        if self.cam_open:
            if len(val) > 1: # have width x height
                self.width  = int(val[0])
                self.height = int(val[1])
            else: # given only one value for resolution
                self.width  = int(val)
                self.height = int(val)
            self._camera_res = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Resolution:{}x{}".format(self._camera_res[0],self._camera_res[1])))
        else: # camera not open
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Resolution, camera not open!"))

    @property
    def exposure(self):
        """ returns curent exposure """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_EXPOSURE)
        else: return float("NaN")
    @exposure.setter
    def exposure(self, val):
        """ # sets current exposure """
        if (val is None):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Exposure to {}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_EXPOSURE, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Exposure set:{}".format(val)))
                    self._exposure = self.cam.get(cv2.CAP_PROP_EXPOSURE)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Exposure is:{}".format(self._exposure)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Expsosure to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Exposure, camera not open!"))

    @property
    def autoexposure(self):
        """ returns curent exposure """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        else: return -1
    @autoexposure.setter
    def autoexposure(self, val):
        """ sets autoexposure """
        if (val is None):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skippingt set Autoexposure to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Autoexposure set:{}".format(val)))
                    self._autoexposure = self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Autoexposure is:{}".format(self._autoexposure)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Autoexposure to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Autoexposure, camera not open!"))

    @property
    def fps(self):
        """ returns current frames per second setting """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_FPS)
        else: return float("NaN")
    @fps.setter
    def fps(self, val):
        """ set frames per second in camera """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set FPS to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_FPS, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FPS set:{}".format(val)))
                    self._framerate = self.cam.get(cv2.CAP_PROP_FPS)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FPS is:{}".format(self._framerate)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FPS to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set FPS, camera not open!"))

    @staticmethod
    def decode_fourcc(val):
        """ decode the fourcc integer to the chracter string """
        return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def fourcc(self):
        """ return video encoding format """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_FOURCC))
        else: return "None"
    @fourcc.setter
    def fourcc(self, val):
        """ set video encoding format in camera """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set FOURCC to:{}".format(val)))
            return
        if self.cam_open:        
            if isinstance(val, str): # fourcc is a string
                with self.cam_lock: 
                    if self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(val[0],val[1],val[2],val[3])):
                        self._fourcc     = self.cam.get(cv2.CAP_PROP_FOURCC)
                        self._fourcc_str = self.decode_fourcc(self._fourcc)
                        if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FOURCC is:{}".format(self._fourcc_str)))
                    else:
                        if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FOURCC to:{}".format(val)))
            else: # fourcc is integer/long
                with self.cam_lock: 
                    if self.cam.set(cv2.CAP_PROP_FOURCC, val):
                        self._fourcc     = int(self.cam.get(cv2.CAP_PROP_FOURCC))
                        self._fourcc_str = self.decode_fourcc(self._fourcc)
                        if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FOURCC is:{}".format(self._fourcc_str)))
                    else:
                        if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FOURCC to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set fourcc, camera not open!"))

    @property
    def buffersize(self):
        """ return opencv camera buffersize """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_BUFFERSIZE))
        else: return float("NaN")
    @buffersize.setter
    def buffersize(self, val):
        """ set opencv camera buffersize """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Buffersize to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_BUFFERSIZE, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Buffersize set:{}".format(val)))
                    self._buffersize = int(self.cam.get(cv2.CAP_PROP_BUFFERSIZE))
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Buffersize is:{}".format(self._buffersize)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Buffersize to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Buffersize, camera not open!"))

    @property
    def gain(self):
        """ return opencv camera gain """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_GAIN))
        else: return float("NaN")
    @gain.setter
    def gain(self, val):
        """ set opencv camera gain """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Gain to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_GAIN, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Gain set:{}".format(val)))
                    self._gain = int(self.cam.get(cv2.CAP_PROP_GAIN))
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Gain is:{}".format(self._gain)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Gain to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Gain, camera not open!"))

    @property
    def wbtemperature(self):
        """ return opencv camera white balance temperature """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_WB_TEMPERATURE))
        else: return float("NaN")
    @wbtemperature.setter
    def wbtemperature(self, val):
        """ set opencv camera white balance temperature """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set WB_TEMPERATURE to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_WB_TEMPERATURE, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:WB_TEMPERATURE set:{}".format(val)))
                    self._wbtemp = int(self.cam.get(cv2.CAP_PROP_WB_TEMPERATURE))
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:WB_TEMPERATURE is:{}".format(self._wbtemp)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set whitebalance temperature to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set whitebalance temperature, camera not open!"))

    @property
    def autowb(self):
        """ return opencv camera auto white balance """
        if self.cam_open:
            return int(self.cam.get(cv2.CAP_PROP_AUTO_WB))
        else: return float("NaN")
    @autowb.setter
    def autowb(self, val):
        """ set opencv camera auto white balance """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set AUTO_WB to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_AUTO_WB, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:AUTO_WB:{}".format(val)))
                    self._autowb = int(self.cam.get(cv2.CAP_PROP_AUTO_WB))
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:AUTO_WB is:{}".format(self._autowb)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set auto whitebalance to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set auto whitebalance, camera not open!"))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    configs = {
        'camera_res'      : (1280, 720 ),   # width & height
        'exposure'        : -2,             # -1,0 = auto, 1...max=frame interval, 
        'autoexposure'    : 1,              # depends on camera: 0.25 or 0.75(auto) or 1(auto), -1, 0
        'fps'             : 30,             # 15, 30, 40, 90, 120, 180
        'fourcc'          : -1,             # n.a.
        'buffersize'      : -1,             # n.a.
        'gain'            : 4,              #
        'wb_temp'         : 4600,           #
        'autowb'          : 1,              #
        'output_res'      : (-1, -1),       # Output resolution, -1,-1 no change
        'flip'            : 0,              # 0=norotation 
                                            # 1=ccw90deg 
                                            # 2=rotation180 
                                            # 3=cw90 
                                            # 4=horizontal 
                                            # 5=upright diagonal flip 
                                            # 6=vertical 
                                            # 7=uperleft diagonal flip
        'displayfps'       : 30             # frame rate for display server
    }

    if configs['displayfps'] >= configs['fps']:  display_interval = 0
    else:                                        display_interval = 1.0/configs['displayfps']

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")
   
    logger.log(logging.DEBUG, "Starting Capture")

    camera = cv2Capture(configs,camera_num=0)     
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")

    window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    last_display = time.perf_counter()

    stop = False  
    while(not stop):
        current_time = time.perf_counter()

        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "{}".format(msg))

        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if (current_time - last_display) >= display_interval:
            cv2.imshow(window_handle, frame)
            last_display = current_time
            try:
                if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(window_handle, 0) < 0): stop = True
            except: 
                stop = True  
         
    camera.stop()
    cv2.destroyAllWindows()
