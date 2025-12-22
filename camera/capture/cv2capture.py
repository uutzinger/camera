###############################################################################
# OpenCV video capture
#
# Uses opencv video capture to capture system's camera
# Adapts to operating system and allows configuration of codec
#
# Urs Utzinger
# 
# Changes:
# 2025 Improved exposure setting robustness across backends
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
import logging, time, sys, math

# Open Computer Vision
import cv2

def as_int(value, default=-1) -> int:
    """Safe int conversion used for OpenCV camera properties.

    OpenCV backends can return NaN/None for unsupported properties; calling int(NaN)
    raises, so we normalize to a sentinel default.
    """
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default


def as_float(value, default=float('nan')) -> float:
    """Safe float conversion used for OpenCV camera properties."""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

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
    def __init__(self, 
        configs: dict, 
        camera_num: int = 0, 
        res: tuple = None,    # width, height
        exposure: float = None,
        queue_size: int = 32
    ):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}
        
        # populate desired settings from configuration file or function arguments
        ####################################################################
        self._camera_num       = camera_num
        if exposure is not None:
            self._exposure    = exposure
        else:
            if 'exposure' in self._configs: self._exposure = self._configs['exposure']
            else:                           self._exposure = -1.0
        if res is not None:
            self._camera_res = res
        else: 
            if 'camera_res' in self._configs: self._camera_res = self._configs['camera_res']
            else:                             self._camera_res = (640, 480)
        if 'output_res' in self._configs:     self._output_res   = self._configs['output_res']
        else:                                 self._output_res   = (-1,-1)
        if 'fps' in self._configs:            self._framerate    = self._configs['fps']
        else:                                 self._framerate    = -1.0
        if 'flip' in self._configs:           self._flip_method  = self._configs['flip']
        else:                                 self._flip_method  = 0
        if 'buffersize' in self._configs:     self._buffersize   = self._configs['buffersize']         # camera drive buffer size
        else:                                 self._buffersize   = 1
        if 'fourcc' in self._configs:         self._fourcc       = self._configs['fourcc']             # camera sensor encoding format
        else:                                 self._fourcc       = -1
        if 'autoexposure' in self._configs:   self._autoexposure = self._configs['autoexposure']       # autoexposure depends on camera
        else:                                 self._autoexposure = -1
        if 'gain' in self._configs:           self._gain         = self._configs['gain']
        else:                                 self._gain         = -1.0
        if 'wb_temp' in self._configs:        self._wbtemp       = self._configs['wb_temp']
        else:                                 self._wbtemp       = -1
        if 'autowb' in self._configs:         self._autowb       = self._configs['autowb']
        else:                                 self._autowb       = -1
        if 'settings' in self._configs:       self._settings     = self._configs['settings']
        else:                                 self._settings     = -1

        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        
        self.capture         = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True
        self.cam_lock        = Lock()

        # open up the camera
        self.open_cam()

        # Init vars
        self.frame_time   = 0.0
        self.measured_fps = 0.0

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        # clean up

    def close_cam(self):
        """Release the underlying VideoCapture (idempotent)."""
        try:
            with self.cam_lock:
                cam = getattr(self, 'cam', None)
                if cam is not None:
                    cam.release()
                self.cam = None
                self.cam_open = False
        except Exception:
            pass

    def start(self):
        """start the capture thread"""
        self.stopped = False
        super().start()

    # Thread entrypoint
    def run(self):
        """Thread entrypoint"""
        if not self.cam_open:
            return

        last_time = time.time()
        num_frames = 0

        try:
            while not self.stopped:
                current_time = time.time()

                if self.cam is None:
                    time.sleep(0.01)
                    continue

                with self.cam_lock:
                    ret, img = self.cam.read()
                if (not ret) or (img is None):
                    time.sleep(0.005)
                    continue

                num_frames += 1
                self.frame_time = int(current_time * 1000)

                if not self.capture.full():
                    img_proc = img

                    # Resize only if an explicit output size was provided
                    if (self._output_width > 0) and (self._output_height > 0):
                        img_proc = cv2.resize(img_proc, self._output_res)

                    # Apply flip/rotation if requested
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

                    self.capture.put_nowait((self.frame_time, img_proc))
                else:
                    if not self.log.full():
                        self.log.put_nowait((logging.WARNING, "CV2:Capture Queue is full!"))

                # FPS calculation
                if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                    self.measured_fps = num_frames/5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "CAM:FPS:{}".format(self.measured_fps)))
                    last_time = current_time
                    num_frames = 0
        finally:
            try:
                self.close_cam()
            except Exception:
                pass

    # Setup the Camera
    ############################################################################
    def open_cam(self):
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
            self.resolution    = self._camera_res      #
            self.fps           = self._framerate       # desired fps
            self.buffersize    = self._buffersize      # camera drive buffer size
            self.fourcc        = self._fourcc          # camera sensor encoding format
            self.gain          = self._gain            # camera gain
            self.wbtemperature = self._wbtemp          # camera white balance temperature
            self.autowb        = self._autowb          # camera enable auto white balance

            # Exposure settings are backend-dependent; apply with robust helper.
            self.apply_exposure_settings()

            if self._settings > -1:
                # OpenCV's CAP_PROP_SETTINGS is backend/OS-specific.
                # It typically opens a native camera property dialog on Windows (e.g. DirectShow).
                # On Linux/V4L2 this usually does nothing.
                ok = self.cam.set(cv2.CAP_PROP_SETTINGS, 0.0)
                if not ok and not self.log.full():
                    self.log.put_nowait((logging.WARNING, "CV2:CAP_PROP_SETTINGS not supported by this backend/OS"))
            
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

    def apply_exposure_settings(self):
        """Apply exposure and autoexposure in a robust, backend-tolerant way.

        Semantics:
        - If self._exposure > 0: request manual exposure (disable AE first), then set exposure.
        - If self._exposure <= 0: do not force exposure; optionally request AE depending on self._autoexposure.
        - If self._autoexposure == -1: do not touch AE.

        Notes:
        - CAP_PROP_AUTO_EXPOSURE values are not portable. We try common candidates and verify via read-back.
        """

        if not self.cam_open:
            return

        try:
            requested_exposure = self._exposure
        except Exception:
            requested_exposure = None

        try:
            requested_ae = self._autoexposure
        except Exception:
            requested_ae = -1

        # Determine desired AE mode.
        manual_requested = requested_exposure is not None and requested_exposure > 0
        if manual_requested:
            desired_ae_mode = 'manual'
        else:
            if requested_ae is None or requested_ae == -1:
                desired_ae_mode = None
            else:
                desired_ae_mode = 'auto' if requested_ae > 0 else 'manual'

        # Apply AE mode first (important when setting manual exposure)
        if desired_ae_mode in ('auto', 'manual'):
            self._try_set_autoexposure(desired_ae_mode)

        # Apply manual exposure only when requested
        if manual_requested:
            ok = self._set_prop(cv2.CAP_PROP_EXPOSURE, requested_exposure)
            readback = self._get_prop(cv2.CAP_PROP_EXPOSURE)
            if ok:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"CV2:Exposure set:{requested_exposure}, readback={readback}"))
            else:
                if not self.log.full():
                    self.log.put_nowait((logging.WARNING, f"CV2:Failed to set Exposure to:{requested_exposure}, readback={readback}"))

    def _try_set_autoexposure(self, mode: str) -> bool:
        """Try to set auto exposure mode using common backend-specific values."""
        if mode not in ('auto', 'manual'):
            return False

        # Candidate values (best-effort) for CAP_PROP_AUTO_EXPOSURE.
        # V4L2 commonly uses 0.25 (manual) / 0.75 (auto), but other backends differ.
        if sys.platform.startswith('linux'):
            manual_candidates = (0.25, 0.0)
            auto_candidates = (0.75, 1.0)
        else:
            manual_candidates = (0.0, 0.25)
            auto_candidates = (1.0, 0.75)

        candidates = auto_candidates if mode == 'auto' else manual_candidates
        before = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)

        for candidate in candidates:
            ok = self._set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, candidate)
            after = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
            if ok:
                # Some drivers don't echo the exact value; accept if it changed or is close.
                try:
                    close = abs(float(after) - float(candidate)) < 1e-3
                except Exception:
                    close = False
                if (after != before) or close:
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"CV2:Autoexposure({mode}) set via {candidate}, readback={after}"))
                    return True

        if not self.log.full():
            self.log.put_nowait((logging.WARNING, f"CV2:Autoexposure({mode}) not supported or rejected (readback={self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)})"))
        return False

    # Open Settings Window
    ############################################################################
    def opensettings(self):
        """
        Open up the camera settings window
        """
        if self.cam_open:
            ok = self.cam.set(cv2.CAP_PROP_SETTINGS, 0.0)
            if not ok and not self.log.full():
                self.log.put_nowait((logging.WARNING, "CV2:CAP_PROP_SETTINGS not supported by this backend/OS"))

    # Camera routines #################################################
    # Reading and setting camera options
    ###################################################################

    @property
    def width(self):
        """ returns video capture width """
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_FRAME_WIDTH), default=-1)
        else: return -1
    @width.setter
    def width(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Width not changed to {}".format(val)))
            return
        if self.cam_open and val > 0:
            if self._set_prop(cv2.CAP_PROP_FRAME_WIDTH, val):
                # HEIGHT and WIDTH only valid if both were set
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Width:{}".format(val)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Width to {}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Width, camera not open!"))

    @property
    def height(self):
        """ returns video capture height """
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_FRAME_HEIGHT), default=-1)
        else: return -1
    @height.setter
    def height(self, val):
        """ sets video capture height """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Height not changed:{}".format(val)))
            return
        if self.cam_open and val > 0:
            if self._set_prop(cv2.CAP_PROP_FRAME_HEIGHT, int(val)):
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
            return (
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_WIDTH), default=-1),
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_HEIGHT), default=-1),
            )
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
            self._camera_res = (
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_WIDTH), default=-1),
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_HEIGHT), default=-1),
            )
            if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Resolution:{}x{}".format(self._camera_res[0],self._camera_res[1])))
        else: # camera not open
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Resolution, camera not open!"))

    @property
    def exposure(self):
        """ returns curent exposure """
        if self.cam_open:
            return self._get_prop(cv2.CAP_PROP_EXPOSURE)
        else: return float("NaN")
    @exposure.setter
    def exposure(self, val):
        """ # sets current exposure """
        if (val is None):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Exposure to {}".format(val)))
            return
        if self.cam_open:
            # If user requests a manual exposure time, try to disable AE first.
            if isinstance(val, (int, float)) and val > 0:
                self._try_set_autoexposure('manual')
            if self._set_prop(cv2.CAP_PROP_EXPOSURE, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Exposure set:{}".format(val)))
                self._exposure = self._get_prop(cv2.CAP_PROP_EXPOSURE)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Exposure is:{}".format(self._exposure)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Expsosure to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Exposure, camera not open!"))

    @property
    def autoexposure(self):
        """ returns current auto exposure setting (backend-specific numeric value) """
        if self.cam_open:
            return self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
        else: return -1
    @autoexposure.setter
    def autoexposure(self, val):
        """sets auto exposure.

        Supported inputs:
        - -1: don't change
        - 0: request manual exposure mode (best-effort)
        - 1: request auto exposure mode (best-effort)
        - any other number: passed through to CAP_PROP_AUTO_EXPOSURE directly
        """
        if (val is None):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Autoexposure to:{}".format(val)))
            return
        if val == -1:
            return
        if self.cam_open:
            # If user passes semantic 0/1, try robust mode setting.
            if val in (0, 1, 0.0, 1.0, True, False):
                mode = 'auto' if bool(val) else 'manual'
                ok = self._try_set_autoexposure(mode)
                self._autoexposure = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
                if not ok and not self.log.full():
                    self.log.put_nowait((logging.WARNING, f"CV2:Autoexposure semantic set({mode}) may not be supported; readback={self._autoexposure}"))
                return

            # Otherwise treat as raw backend-specific value.
            if self._set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Autoexposure set:{}".format(val)))
                self._autoexposure = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Autoexposure is:{}".format(self._autoexposure)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Autoexposure to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Autoexposure, camera not open!"))

    @property
    def fps(self):
        """ returns current frames per second setting """
        if self.cam_open:
            return self._get_prop(cv2.CAP_PROP_FPS)
        else: return float("NaN")
    @fps.setter
    def fps(self, val):
        """ set frames per second in camera """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set FPS to:{}".format(val)))
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_FPS, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FPS set:{}".format(val)))
                self._framerate = self._get_prop(cv2.CAP_PROP_FPS)
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
            return as_int(self._get_prop(cv2.CAP_PROP_FOURCC), default=-1)
        else: return "None"
    @fourcc.setter
    def fourcc(self, val):
        """ set video encoding format in camera """
        if (val is None) or (val == -1):
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set FOURCC to:{}".format(val)))
            return
        if self.cam_open:        
            if isinstance(val, str): # fourcc is a string
                if self._set_prop(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(val[0],val[1],val[2],val[3])):
                    self._fourcc     = self._get_prop(cv2.CAP_PROP_FOURCC)
                    self._fourcc_str = self.decode_fourcc(self._fourcc)
                    if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:FOURCC is:{}".format(self._fourcc_str)))
                else:
                    if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set FOURCC to:{}".format(val)))
            else: # fourcc is integer/long
                if self._set_prop(cv2.CAP_PROP_FOURCC, val):
                    self._fourcc     = as_int(self._get_prop(cv2.CAP_PROP_FOURCC), default=-1)
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
            return as_int(self._get_prop(cv2.CAP_PROP_BUFFERSIZE), default=-1)
        else: return float("NaN")
    @buffersize.setter
    def buffersize(self, val):
        """ set opencv camera buffersize """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Buffersize to:{}".format(val)))
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_BUFFERSIZE, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Buffersize set:{}".format(val)))
                self._buffersize = as_int(self._get_prop(cv2.CAP_PROP_BUFFERSIZE), default=-1)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Buffersize is:{}".format(self._buffersize)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Buffersize to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Buffersize, camera not open!"))

    @property
    def gain(self):
        """ return opencv camera gain """
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_GAIN), default=-1)
        else: return float("NaN")
    @gain.setter
    def gain(self, val):
        """ set opencv camera gain """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set Gain to:{}".format(val)))
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_GAIN, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Gain set:{}".format(val)))
                self._gain = as_int(self._get_prop(cv2.CAP_PROP_GAIN), default=-1)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:Gain is:{}".format(self._gain)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set Gain to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set Gain, camera not open!"))

    @property
    def wbtemperature(self):
        """ return opencv camera white balance temperature """
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_WB_TEMPERATURE), default=-1)
        else: return float("NaN")
    @wbtemperature.setter
    def wbtemperature(self, val):
        """ set opencv camera white balance temperature """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set WB_TEMPERATURE to:{}".format(val)))
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_WB_TEMPERATURE, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:WB_TEMPERATURE set:{}".format(val)))
                self._wbtemp = as_int(self._get_prop(cv2.CAP_PROP_WB_TEMPERATURE), default=-1)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:WB_TEMPERATURE is:{}".format(self._wbtemp)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set whitebalance temperature to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set whitebalance temperature, camera not open!"))

    @property
    def autowb(self):
        """ return opencv camera auto white balance """
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_AUTO_WB), default=-1)
        else: return float("NaN")
    @autowb.setter
    def autowb(self, val):
        """ set opencv camera auto white balance """
        if val is None or val < 0:
            if not self.log.full(): self.log.put_nowait((logging.WARNING, "CV2:Skipping set AUTO_WB to:{}".format(val)))
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_AUTO_WB, val):
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:AUTO_WB:{}".format(val)))
                self._autowb = as_int(self._get_prop(cv2.CAP_PROP_AUTO_WB), default=-1)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "CV2:AUTO_WB is:{}".format(self._autowb)))
            else:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "CV2:Failed to set auto whitebalance to:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "CV2:Failed to set auto whitebalance, camera not open!"))

    def _set_prop(self, prop_id, value) -> bool:
        if not self.cam_open:
            return False
        with self.cam_lock:
            try:
                return bool(self.cam.set(prop_id, value))
            except Exception:
                return False

    def _get_prop(self, prop_id):
        if not self.cam_open:
            return float('nan')
        with self.cam_lock:
            try:
                return self.cam.get(prop_id)
            except Exception:
                return float('nan')

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

    window_name = "Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    last_display = time.perf_counter()

    stop = False  
    while(not stop):
        current_time = time.perf_counter()

        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "{}".format(msg))

        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if (current_time - last_display) >= display_interval:
            cv2.imshow(window_name, frame)
            last_display = current_time
            try:
                if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(window_name, 0) < 0): stop = True
            except: 
                stop = True  
         

    camera.stop()
    cv2.destroyAllWindows()
