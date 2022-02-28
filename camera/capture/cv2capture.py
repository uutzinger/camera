###############################################################################
# OpenCV video capture
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
# 
# 2021 Initialize, Remove Frame acces (use only queue)
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
            self._exposure   = configs['exposure']
        if res is not None:
            self._camera_res = res
        else: 
            self._camera_res = configs['camera_res']
        self._output_res     = configs['output_res']
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._framerate      = configs['fps']
        self._flip_method    = configs['flip']
        self._buffersize     = configs['buffersize']         # camera drive buffer size
        self._fourcc         = configs['fourcc']             # camera sensor encoding format
        self._autoexposure   = configs['autoexposure']       # autoexposure depends on camera

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
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_MSMF)
        elif sys.platform.startswith('darwin'):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('linux'):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_V4L2)
        else:
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_ANY)

        self.cam_open = self.cam.isOpened()

        if self.cam_open:
            # Apply settings to camera
            self.height        = self._camera_res[1]   # image resolution
            self.width         = self._camera_res[0]   # image resolution
            self.autoexposure  = self._autoexposure    # autoexposure
            self.exposure      = self._exposure        # camera exposure
            self.fps           = self._framerate       # desired fps
            self.buffersize    = self._buffersize      # camera drive buffer size
            self.fourcc        = self._fourcc          # camera sensor encoding format
            # Update records
            self._camera_res   = self.resolution
            self._exposure     = self.exposure
            self._buffersize   = self.buffersize
            self._framerate    = self.fps
            self._autoexposure = self.autoexposure
            self._fourcc       = self.fourcc
            self._fourcc_str   = self.decode_fourcc(self._fourcc)
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to open camera!"))

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
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set width to {}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set width, camera not open!"))

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
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set height to {}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set height, camera not open!"))

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
        else: # camera not open
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set resolution, camera not open!"))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set exposure to {}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_EXPOSURE, val):
                    self._exposure = self.cam.get(cv2.CAP_PROP_EXPOSURE)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Exposure:{}".format(self.exposure)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set expsosure to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set exposure, camera not open!"))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set Autoexposure to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                    self._autoexposure = self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Autoexposure:{}".format(self.autoexposure)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Autoexposure to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set auto exposure, camera not open!"))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set framerate to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_FPS, val):
                    self._framerate = self.cam.get(cv2.CAP_PROP_FPS)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FPS:{}".format(self.fps)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FPS to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set framerate, camera not open!"))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set FOURCC to:{}".format(val)))
            return
        if self.cam_open:        
            if isinstance(val, str): # fourcc is a string
                with self.cam_lock: 
                    if self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(val[0],val[1],val[2],val[3])):
                        self._fourcc     = self.cam.get(cv2.CAP_PROP_FOURCC)
                        self._fourcc_str = self.decode_fourcc(self._fourcc)
                        if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FOURCC:{}".format(self._fourcc_str)))
                    else:
                        if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FOURCC to:{}".format(val)))
            else: # fourcc is integer/long
                with self.cam_lock: 
                    if self.cam.set(cv2.CAP_PROP_FOURCC, val):
                        self._fourcc     = int(self.cam.get(cv2.CAP_PROP_FOURCC))
                        self._fourcc_str = self.decode_fourcc(self._fourcc)
                        if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FOURCC:{}".format(self._fourcc_str)))
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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set buffer size to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                if self.cam.set(cv2.CAP_PROP_BUFFERSIZE, val):
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Buffersize:{}".format(val)))
                    self._buffersize = int(self.cam.get(cv2.CAP_PROP_BUFFERSIZE))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set buffer size to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set buffersize, camera not open!"))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    configs = {
        'camera_res'      : (1920,1080),      # 
        'exposure'        : 1,              # 
        'autoexposure'    : 0,              # 
        'fps'             : 30,             # 
        'fourcc'          : 'MJPG',         # 
        'buffersize'      : -1,             #  
        'output_res'      : (-1, -1),       #  
        'flip'            : 0,              #  
        'displayfps'       : 10             # 
    }

    if configs['displayfps'] >= configs['fps']:  display_interval = 0
    else:                                        display_interval = 1.0/configs['displayfps']

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")
   
    logger.log(logging.DEBUG, "Starting Capture")

    camera = cv2Capture(configs,camera_num=1)     
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
            cv2.imshow('Camera', frame)
            last_display = current_time
            if cv2.waitKey(1) & 0xFF == ord('q'):  stop=True
            #try: 
            #    if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0: 
            #        stop = True
            #except: 
            #    stop = True
         

    camera.stop()
    cv2.destroyAllWindows()
