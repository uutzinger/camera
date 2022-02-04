###############################################################################
# OpenCV video capture
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
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
        exposure: float = None):
        
        # populate desired settings from configuration file or function arguments
        ####################################################################
        self.camera_num       = camera_num
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

        self.capture         = Queue(maxsize=32)
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
                    if (not self.log.full()): self.log.put_nowait((logging.WARNING, "CV2:Capture Queue is full!"))


            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if (not self.log.full()): self.log.put_nowait((logging.INFO, "CAM:FPS:{}".format(self.measured_fps)))
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
            self.cam = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_MSMF)
        elif sys.platform.startswith('darwin'):
            self.cam = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('linux'):
            self.cam = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_V4L2)
        else:
            self.cam = cv2.VideoCapture(self.camera_num, apiPreference=cv2.CAP_ANY)

        self.cam_open = self.cam.isOpened()

        if self.cam_open:
            # Apply settings to camera
            if self._camera_res[0] > 0:
                self.width          = self._camera_res[0]       # image resolution
            if self._camera_res[1] > 0:
                self.height         = self._camera_res[1]       # image resolution
            self.autoexposure   = self._autoexposure            # autoexposure
            if self._exposure > 0:
                self.exposure   = self._exposure                # camera exposure
            if self._buffersize > 0:
                self.buffersize = self._buffersize              # camera drive buffer size
            if not self._fourcc == -1:
                self.fourcc     = self._fourcc                  # camera sensor encoding format
            if self._framerate > 0:
                self.fps        = self._framerate               # desired fps
        else:
            self.log.put_nowait((logging.CRITICAL, "CV2:Failed to open camera!"))

    # Camera routines #################################################
    # Reading and setting camera options
    ###################################################################

    @property
    def width(self):
        """ returns video capture width """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        else: return float("NaN")

    @width.setter
    def width(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            self.log.put_nowait((logging.WARNING, "CV2:Width not changed:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock: 
                isok = self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, val)
            if isok:
                self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set width to {}!".format(val)))

    @property
    def height(self):
        """ returns videocapture height """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else: return float("NaN")

    @height.setter
    def height(self, val):
        """ sets video capture height """
        if (val is None) or (val == -1):
            self.log.put_nowait((logging.WARNING, "CV2:Height not changed:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock: 
                isok = self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val))
            if isok:
                self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set height to {}!".format(val)))

    @property
    def resolution(self):
        """ returns current resolution width x height """
        if self.cam_open:
            return [self.cam.get(cv2.CAP_PROP_FRAME_WIDTH), 
                    self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        else: return [float("NaN"), float("NaN")] 

    @resolution.setter
    def resolution(self, val):
        if val is None: return
        if self.cam_open:
            if len(val) > 1: # have width x height
                with self.cam_lock: 
                    isok0 = self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(val[0]))
                    isok1 = self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val[1]))
                if isok0 and isok1:
                    self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val[0])))
                    self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val[1])))
                else:
                    self.log.put_nowait((logging.ERROR, "CV2:Failed to set resolution to {},{}!".format(val[0],val[1])))
            else: # given only one value for resolution
                with self.cam_lock: 
                    isok0 = self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(val))
                    isok1 = self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val))
                if isok0 and isok1:
                    self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val)))
                    self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val)))
                else:
                    self.log.put_nowait((logging.ERROR, "CV2:Failed to set resolution to {},{}!".format(val,val)))
        else: # camera not open
            self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set resolution, camera not open!"))

    @property
    def exposure(self):
        """ returns curent exposure """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_EXPOSURE)
        else: return float("NaN")

    @exposure.setter
    def exposure(self, val):
        """ # sets current exposure """
        self._exposure = val
        if (val is None) or (val == -1):
            self.log.put_nowait((logging.WARNING, "CV2:Can not set exposure to {}!".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                isok = self.cam.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
            if isok:
                self.log.put_nowait((logging.INFO, "CV2:Exposure:{}".format(val)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set expsosure to:{}".format(val)))

    @property
    def autoexposure(self):
        """ returns curent exposure """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        else: return float("NaN")

    @autoexposure.setter
    def autoexposure(self, val):
        """ sets autoexposure """
        if (val is None) or (val == -1):
            self.log.put_nowait((logging.WARNING, "CV2:Can not set Autoexposure to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                isok = self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            if isok:
                self.log.put_nowait((logging.INFO, "CV2:Autoexposure:{}".format(val)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set Autoexposure to:{}".format(val)))

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
            self.log.put_nowait((logging.WARNING, "CV2:Can not set framerate to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                isok = self.cam.set(cv2.CAP_PROP_FPS, val)
            if isok:
                self.log.put_nowait((logging.INFO, "CV2:FPS:{}".format(val)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set FPS to:{}".format(val)))

    @staticmethod
    def decode_fourcc(val):
        """ decode the fourcc integer to the chracter string """
        return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def fourcc(self):
        """ return video encoding format """
        if self.cam_open:
            self._fourcc = self.cam.get(cv2.CAP_PROP_FOURCC)
            self._fourcc_str = self.decode_fourcc(self._fourcc)
            return self._fourcc_str
        else: return "None"

    @fourcc.setter
    def fourcc(self, val):
        """ set video encoding format in camera """
        if (val is None) or (val == -1):
            self.log.put_nowait((logging.WARNING, "CV2:Can not set FOURCC to:{}!".format(val)))
            return
        if isinstance(val, str):  # we need to convert from FourCC to integer
            self._fourcc     = cv2.VideoWriter_fourcc(val[0],val[1],val[2],val[3])
            self._fourcc_str = val
        else: # fourcc is integer/long
            self._fourcc     = val
            self._fourcc_str = self.decode_fourcc(val)
        if self.cam_open:
            with self.cam_lock: 
                isok = self.cam.set(cv2.CAP_PROP_FOURCC, self._fourcc)
            if isok :
                self.log.put_nowait((logging.INFO, "CV2:FOURCC:{}".format(self._fourcc_str)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set FOURCC to:{}".format(self._fourcc_str)))

    @property
    def buffersize(self):
        """ return opencv camera buffersize """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_BUFFERSIZE)
        else: return float("NaN")

    @buffersize.setter
    def buffersize(self, val):
        """ set opencv camera buffersize """
        if val is None or val == -1:
            self.log.put_nowait((logging.WARNING, "CV2:Can not set buffer size to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                isok = self.cam.set(cv2.CAP_PROP_BUFFERSIZE, val)
            if isok:
                self.log.put_nowait((logging.INFO, "CV2:Buffersize:{}".format(val)))
            else:
                self.log.put_nowait((logging.ERROR, "CV2:Failed to set buffer size to:{}".format(val)))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    configs = {
        'camera_res'      : (1920, 1080),   # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
        'exposure'        : -6,             # camera specific e.g. -5 =(2^-5)=1/32, 0 = auto, 1...max=frame interval in microseconds
        'autoexposure'    : 3.0,            # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
        'fps'             : 30,             # 120fps only with MJPG fourcc
        'fourcc'          : "MJPG",         # cv2 camera only: MJPG, YUY2, YUYV
        'buffersize'      : -1,             # default is 4 for V4L2, max 10, 
        'fov'             : 77,             # camera lens field of view in degress
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 0,              # 0=norotation 
        'displayfps'       : 5              # frame rate for display server
    }

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")
   
    logger.log(logging.DEBUG, "Starting Capture")

    camera = cv2Capture(configs,camera_num=0)     
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")

    window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty("Camera", 0) >= 0):
        try:
            (frame_time, frame) = camera.capture.get()
            cv2.imshow('Camera', frame)
        except: pass

        if cv2.waitKey(1) & 0xFF == ord('q'):  break

        try: 
            (level, msg)=camera.log.get_nowait()
            logger.log(level, "CV2:{}".format(msg))
        except: pass

    camera.stop()
    cv2.destroyAllWindows()
