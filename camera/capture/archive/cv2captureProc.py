###############################################################################
# OpenCV video capture
# 
# Attempting implementation with multiprocessing
#
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
#
# 2021 Initial release, based on threaded version
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Processing
from multiprocessing import Process, Event, Queue, Lock

# Queue exceptions
import queue

# System
import sys, time, logging

# Open Computer Vision
import cv2

###############################################################################
# Video Capture
###############################################################################

class cv2CaptureProc(Process):
    """
    This process continually captures frames from camera
    """
    # Initialize the camera process
    def __init__(self, configs, 
        camera_num: int = 0, 
        res: tuple = None,    # width, height
        exposure: float = None,
        queue_size: int = 32):

        # Proper Process initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}

        # Process synchronization, events, queues
        self.stopper  = Event()
        self.log      = Queue(maxsize=queue_size)
        self.capture  = Queue(maxsize=queue_size)
        self.cam_lock = Lock()

        # Camera state (in the child process)
        self.cam = None
        self.cam_open = False

        # populate desired settings from configuration file or function arguments
        ####################################################################
        self.camera_num       = camera_num
        if exposure is not None:
            self._exposure    = exposure  
        else: 
            self._exposure   = self._configs.get('exposure', -1)
        if res is not None:
            self._camera_res = res
        else: 
            self._camera_res = self._configs.get('camera_res', (640, 480))
        self._output_res      = self._configs.get('output_res', (-1, -1))
        self._output_width    = self._output_res[0]
        self._output_height   = self._output_res[1]
        self._framerate       = self._configs.get('fps', -1)
        self._flip_method     = self._configs.get('flip', 0)
        self._buffersize      = self._configs.get('buffersize', 1)         # camera drive buffer size
        self._fourcc          = self._configs.get('fourcc', -1)             # camera sensor encoding format
        self._autoexposure    = self._configs.get('autoexposure', -1)       # autoexposure depends on camera

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """Stop the process"""
        self.stopper.set()
        try:
            if self.is_alive():
                self.join(timeout=5.0)
        finally:
            # Best-effort cleanup of IPC resources (parent side)
            try:
                self.log.close()
            except Exception:
                pass
            try:
                self.capture.close()
            except Exception:
                pass

    def close_cam(self):
        """Release the underlying VideoCapture (idempotent, best-effort)."""
        try:
            with self.cam_lock:
                cam = getattr(self, 'cam', None)
                if cam is not None:
                    cam.release()
                self.cam = None
                self.cam_open = False
        except Exception:
            pass

    def start(self, capture_queue = None):
        """Start the capture process.

        Note: capture_queue is kept only for backward compatibility.
        """
        self.stopper.clear()
        super().start()

    # Process entrypoint
    def run(self):
        """Process entrypoint."""

        # open up the camera
        self._open_capture()

        if not self.cam_open:
            return

        # initialize variables
        last_time = time.time()
        num_frames = 0

        try:
            while not self.stopper.is_set():

                # Capture image
                current_time = time.time()

                with self.cam_lock:
                    cam = self.cam
                    if cam is None:
                        time.sleep(0.01)
                        continue
                    ret, img = cam.read()

                if (not ret) or (img is None):
                    time.sleep(0.005)
                    continue

                num_frames += 1

                img_proc = img

                # Resize only if an explicit output size was provided
                if (self._output_width > 0) and (self._output_height > 0):
                    img_proc = cv2.resize(img_proc, self._output_res)

                # Apply flip/rotation if requested (same enum as cv2Capture)
                if self._flip_method != 0:
                    if self._flip_method == 1: # ccw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 2: # rot 180
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_180)
                    elif self._flip_method == 3: # cw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_CLOCKWISE)
                    elif self._flip_method == 4: # horizontal (left-right)
                        img_proc = cv2.flip(img_proc, 1)
                    elif self._flip_method == 5: # upright diagonal. ccw & lr
                        img_proc = cv2.flip(cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                    elif self._flip_method == 6: # vertical (up-down)
                        img_proc = cv2.flip(img_proc, 0)
                    elif self._flip_method == 7: # upperleft diagonal
                        img_proc = cv2.transpose(img_proc)

                try:
                    self.capture.put_nowait((current_time * 1000.0, img_proc))
                except queue.Full:
                    try:
                        if not self.log.full():
                            self.log.put_nowait((logging.WARNING, "CV2:Capture queue is full!"))
                    except Exception:
                        pass

                # FPS calculation
                if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                    try:
                        if not self.log.full():
                            self.log.put_nowait((logging.INFO, "CV2CAM:FPS:{}".format(num_frames/5.0)))
                    except Exception:
                        pass
                    num_frames = 0
                    last_time = current_time
        finally:
            self.close_cam()

    def _open_capture(self):
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
                self.width      = self._camera_res[0]  # image resolution
            if self._camera_res[1] > 0:
                self.height     = self._camera_res[1]  # image resolution
            self.autoexposure   = self._autoexposure   # autoexposure
            if self._exposure > 0:
                self.exposure   = self._exposure       # camera exposure
            if self._buffersize > 0:
                self.buffersize = self._buffersize     # camera drive buffer size
            if not self._fourcc == -1:
                self.fourcc     = self._fourcc         # camera sensor encoding format
            if self._framerate > 0:
                self.fps        = self._framerate      # desired fps
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to open camera!"))

    #
    # Camera routines
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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Width not changed:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock: 
                isok = self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, val)
            if isok:
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set width to {}!".format(val)))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Height not changed:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock: 
                isok = self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val))
            if isok:
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set height to {}!".format(val)))

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
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val[0])))
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val[1])))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set resolution to {},{}!".format(val[0],val[1])))
            else: # given only one value for resolution
                with self.cam_lock: 
                    isok0 = self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(val))
                    isok1 = self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val))
                if isok0 and isok1:
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val)))
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Height:{}".format(val)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set resolution to {},{}!".format(val,val)))
        else: # camera not open
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set resolution, camera not open!"))

    @property
    def exposure(self):
        """ returns current exposure """
        if self.cam_open:
            return self.cam.get(cv2.CAP_PROP_EXPOSURE)
        else: return float("NaN")

    @exposure.setter
    def exposure(self, val):
        """ sets current exposure """
        self._exposure = val
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set exposure to {}!".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                isok = self.cam.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
            if isok:
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Exposure:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set expsosure to:{}".format(val)))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set Autoexposure to:{}".format(val)))
            return
        if self.cam_open:
            with self.cam_lock:
                isok = self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            if isok:
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Autoexposure:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Autoexposure to:{}".format(val)))

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
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FPS:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FPS to:{}".format(val)))

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
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Can not set FOURCC to:{}!".format(val)))
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
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FOURCC:{}".format(self._fourcc_str)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FOURCC to:{}".format(self._fourcc_str)))

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
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Buffersize:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set buffer size to:{}".format(val)))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")

    logger.log(logging.DEBUG, "Starting Capture")

    configs = {
        'exposure': -1,
        'camera_res': (640, 480),
        'output_res': (-1, -1),
        'fps': -1,
        'flip': 0,
        'buffersize': 1,
        'fourcc': -1,
        'autoexposure': -1,
    }

    camera = cv2CaptureProc(configs, camera_num=0, res=(640, 480), exposure=-1)
    camera.start()

    logger.log(logging.INFO, "Getting Frames")
    window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty("Camera", 0) >= 0):
        try:
            (frame_time, frame) = camera.capture.get()
            cv2.imshow('Camera', frame)
            # print(camera.frame.shape)
        except: pass
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        try: 
            (level, msg)=camera.log.get_nowait()
            logger.log(level, "CV2:{}".format(msg))
        except: pass

    camera.stop()
