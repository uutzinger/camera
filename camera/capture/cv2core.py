###############################################################################
# OpenCV video capture core
#
# Non-threaded core used by threaded and Qt wrappers.
#
# Urs Utzinger
###############################################################################
# Changes:
# 2026 Core, qt wrapper, non-qt wrapper
# 2025 Improved exposure setting robustness across backends
# 2022 Added access to more opencv camera properties
#      Auto populates missing configs
#      Access to opencv camera configs window
# 2021 Initialize, Remove direct Frame acces (use only queue)
# 2019 Initial release, based on Bitbuckets FRC 4183 code
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: cv2Core
# This is a non-threaded, non-Qt core wrapper intended to be composed by:
# - the threaded capture wrapper `piCamera2Capture`
# - the Qt wrapper `piCamera2CaptureQt`
#
# Public methods:
# - open_cam()         : Open and configure the camera, allocate frame buffer.
# - close_cam()        : Stop and close the camera, release frame buffer.
# - capture_array()    : tuple[np.ndarray | None, float | None]
#                        Capture a frame and timestamp from the configured stream.
# - get_control(name: str) -> Any
#                        Get a single control/metadata value.
# - set_controls(controls: dict) -> bool
#                        Set multiple controls on the camera.
# - get_supported_main_color_formats() -> list[str]
# - get_supported_raw_color_formats() -> list[str]
# - get_supported_raw_options() -> list[dict]
# - get_supported_main_options() -> list[dict]
# - log_stream_options() -> None
#
# Convenience properties:
# - cam_open: bool         (camera open state)
# - capturing: bool        (capture state flag)
# - buffer: FrameBuffer    (allocated frame buffer, or None if closed)
# - metadata: dict         (latest frame metadata)
# - width / height / resolution / size
#       (changing size/flip while capturing is discouraged;
#        wrappers should ensure capture is paused first)
# - exposure, gain, autoexposure, aemeteringmode, aeexposuremode
# - autowb, awbmode, wbtemperature, colourgains
# - afmode, lensposition
# - brightness, contrast, saturation, sharpness, noisereductionmode
# - fps
#
# Supported config parameters (configs dict):
# -------------------------------------------
#
# Core capture / output:
# - camera_res: tuple[int,int]      Main stream capture size (w,h)
# - output_res: tuple[int,int]      If >0, prefer libcamera scaling for main;
#                                   in raw mode, may trigger CPU resize
# - fps: float|int                  Requested framerate
# - low_latency: bool               Hint to reduce buffering (core may request fewer libcamera buffers)
# - buffersize: int                 FrameBuffer capacity (default: 32)
# - buffer_overwrite: bool          If True, FrameBuffer overwrites oldest frames when full
#
# Stream selection:
# - mode: str                       'main' (processed) or 'raw' (sensor Bayer)
# - stream_policy: str              'default'|'maximize_fov'|'maximize_fps'
#
# Formats:
# - format: str                     Legacy/combined request (e.g. 'BGR3', 'YUY2', 'SRGGB8')
# - fourcc: str                     Legacy FOURCC alternative
# - main_format: str                Explicit main stream format override
# - raw_format: str                 Explicit raw Bayer format override
# - raw_res: tuple[int,int]         Requested raw sensor window size (w,h)
#
# Controls (applied at open, and/or via properties/set_controls):
# - exposure: int|float             Manual exposure in microseconds; <=0 leaves AE enabled
# - autoexposure: int               -1 leave unchanged, 0 AE off, 1 AE on
# - aemeteringmode: int|str         0/1/2 or 'center'|'spot'|'matrix'
# - autowb: int                     -1 leave unchanged, 0 AWB off, 1 AWB on
# - awbmode: int|str                0..5 or 'auto'|'tungsten'|'fluorescent'|'indoor'|'daylight'|'cloudy'
#
# Low-level / advanced:
# - buffer_count: int|None          Requested libcamera buffer_count (if supported)
# - hw_transform: bool              Attempt libcamera hardware Transform for flip/rotation
# - flip: int                       0..7 (same enum as cv2Capture)
# - test_pattern: bool|str          Enable synthetic frame mode (for testing)
###############################################################################

from __future__ import annotations

from threading import Lock
from typing import Optional, Tuple
from queue import Queue # logging
import logging
import math
import sys

import cv2

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .framebuffer import FrameBuffer

# Static Functions

def as_int(value, default=-1) -> int:
    """Safe int conversion used for OpenCV camera properties."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return int(value)
    except Exception:
        return default

def as_float(value, default=float("nan")) -> float:
    """Safe float conversion used for OpenCV camera properties."""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

class cv2Core:
    """
    Non-threaded OpenCV capture core.

    Public methods:
    - open_cam()
    - close_cam()
    - apply_exposure_settings()
    - opensettings()

    Properties:
    - width, height, resolution
    - exposure, autoexposure
    - fps, fourcc, buffersize
    - gain, wbtemperature, autowb
    """

    def __init__(
        self,
        configs: dict,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        log_queue: Queue | None = None,
    ) -> None:

        self._configs = configs or {}
        self._camera_num = int(camera_num)
        self.log = log_queue

        # Populate desired settings from configuration file or function arguments
        if exposure is not None:
            self._exposure = exposure
        else:
            self._exposure = self._configs.get("exposure", -1.0)

        if res is not None:
            self._camera_res = res
        else:
            self._camera_res = self._configs.get("camera_res", (640, 480))

        self._output_res = self._configs.get("output_res", (-1, -1))
        self._framerate = self._configs.get("fps", -1.0)
        self._flip_method = self._configs.get("flip", 0)
        self._buffersize = self._configs.get("buffersize", 1)
        self._fourcc = self._configs.get("fourcc", -1)
        self._autoexposure = self._configs.get("autoexposure", -1)
        self._gain = self._configs.get("gain", -1.0)
        self._wbtemp = self._configs.get("wb_temp", -1)
        self._autowb = self._configs.get("autowb", -1)
        self._settings = self._configs.get("settings", -1)

        self._output_width = self._output_res[0]
        self._output_height = self._output_res[1]

        self.cam = None
        self.cam_open = False
        self.cam_lock = Lock()

        Will need to add test patter and opencv has some nice one
        # Synthetic/test pattern mode (bypasses Picamera2)
        # - False/None: normal camera
        # - True: enable with default pattern
        # - str: pattern name ('gradient', 'checker', 'noise', 'static')
        self._test_pattern = self._configs.get("test_pattern", False)
        self._test_frame: np.ndarray | None = None
        self._test_next_t: float = 0.0

        # Conversion caching
        self._convert_format = "BGR888"  # Default target format
        self._cv2_conversion_code = None
        self._needs_bit_depth_conversion = False
        self._bit_depth_shift = 0

        # Frame Buffer management
        self._buffer_capacity = int(self._configs.get("buffersize", 32))
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        # not used here self._buffer_copy_on_pull = bool(self._configs.get("buffer_copy", False))
        self._buffer = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_array(self) -> tuple[np.ndarray | None, float | None]:
        pass
        # should have 
        # obtain frame
        # create or obtain time stamp
        # color convert
        # qlpha channel remove
        # CPU resize
        # CPU flip and rotation

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str) -> None:
        if self.log is None:
            return
        try:
            if not self.log.full():
                self.log.put_nowait((level, message))
        except Exception:
            pass

    def open_cam(self) -> None:
        """Open the camera and apply configuration properties."""
        if sys.platform.startswith("win"):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_DSHOW)
        elif sys.platform.startswith("darwin"):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith("linux"):
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_V4L2)
        else:
            self.cam = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_ANY)

        self.cam_open = bool(self.cam.isOpened()) if self.cam is not None else False

        if not self.cam_open:
            self._log(logging.CRITICAL, "CV2:Failed to open camera!")
            return

        # Apply settings to camera
        self.resolution = self._camera_res
        self.fps = self._framerate
        self.buffersize = self._buffersize
        self.fourcc = self._fourcc
        self.gain = self._gain
        self.wbtemperature = self._wbtemp
        self.autowb = self._autowb

        # Exposure settings are backend-dependent; apply with robust helper.
        self.apply_exposure_settings()

        if self._settings > -1:
            ok = self._set_prop(cv2.CAP_PROP_SETTINGS, 0.0)
            if not ok:
                self._log(logging.WARNING, "CV2:CAP_PROP_SETTINGS not supported by this backend/OS")

        # Update records
        self._camera_res = self.resolution
        self._exposure = self.exposure
        self._buffersize = self.buffersize
        self._framerate = self.fps
        self._autoexposure = self.autoexposure
        self._fourcc = self.fourcc
        self._fourcc_str = self.decode_fourcc(self._fourcc)
        self._gain = self.gain
        self._wbtemperature = self.wbtemperature
        self._autowb = self.autowb

    def close_cam(self) -> None:
        """Release the underlying VideoCapture (idempotent)."""
        try:
            with self.cam_lock:
                cam = self.cam
                if cam is not None:
                    cam.release()
                self.cam = None
                self.cam_open = False
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Exposure helpers
    # ------------------------------------------------------------------

    def apply_exposure_settings(self) -> None:
        """Apply exposure and autoexposure in a robust, backend-tolerant way."""
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
            desired_ae_mode = "manual"
        else:
            if requested_ae is None or requested_ae == -1:
                desired_ae_mode = None
            else:
                desired_ae_mode = "auto" if requested_ae > 0 else "manual"

        # Apply AE mode first (important when setting manual exposure)
        if desired_ae_mode in ("auto", "manual"):
            self._try_set_autoexposure(desired_ae_mode)

        # Apply manual exposure only when requested
        if manual_requested:
            ok = self._set_prop(cv2.CAP_PROP_EXPOSURE, requested_exposure)
            readback = self._get_prop(cv2.CAP_PROP_EXPOSURE)
            if ok:
                self._log(logging.INFO, f"CV2:Exposure set:{requested_exposure}, readback={readback}")
            else:
                self._log(logging.WARNING, f"CV2:Failed to set Exposure to:{requested_exposure}, readback={readback}")

    def _try_set_autoexposure(self, mode: str) -> bool:
        """Try to set auto exposure mode using common backend-specific values."""
        if mode not in ("auto", "manual"):
            return False

        if sys.platform.startswith("linux"):
            manual_candidates = (0.25, 0.0)
            auto_candidates = (0.75, 1.0)
        else:
            manual_candidates = (0.0, 0.25)
            auto_candidates = (1.0, 0.75)

        candidates = auto_candidates if mode == "auto" else manual_candidates
        before = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)

        for candidate in candidates:
            ok = self._set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, candidate)
            after = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
            if ok:
                try:
                    close = abs(float(after) - float(candidate)) < 1e-3
                except Exception:
                    close = False
                if (after != before) or close:
                    self._log(logging.INFO, f"CV2:Autoexposure({mode}) set via {candidate}, readback={after}")
                    return True

        self._log(
            logging.WARNING,
            f"CV2:Autoexposure({mode}) not supported or rejected (readback={self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)})",
        )
        return False

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------

    def opensettings(self) -> None:
        """Open up the camera settings window (best-effort)."""
        if self.cam_open:
            ok = self._set_prop(cv2.CAP_PROP_SETTINGS, 0.0)
            if not ok:
                self._log(logging.WARNING, "CV2:CAP_PROP_SETTINGS not supported by this backend/OS")

    # ------------------------------------------------------------------
    # Camera properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_FRAME_WIDTH), default=-1)
        return -1

    @width.setter
    def width(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"CV2:Width not changed to {val}")
            return
        if self.cam_open and val > 0:
            if self._set_prop(cv2.CAP_PROP_FRAME_WIDTH, val):
                self._log(logging.INFO, f"CV2:Width:{val}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set Width to {val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Width, camera not open!")

    @property
    def height(self) -> int:
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_FRAME_HEIGHT), default=-1)
        return -1

    @height.setter
    def height(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"CV2:Height not changed:{val}")
            return
        if self.cam_open and val > 0:
            if self._set_prop(cv2.CAP_PROP_FRAME_HEIGHT, int(val)):
                self._log(logging.INFO, f"CV2:Height:{val}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set Height to {val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Height, camera not open!")

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cam_open:
            return (
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_WIDTH), default=-1),
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_HEIGHT), default=-1),
            )
        return (-1, -1)

    @resolution.setter
    def resolution(self, val) -> None:
        if val is None:
            return
        if self.cam_open:
            if len(val) > 1:
                self.width = int(val[0])
                self.height = int(val[1])
            else:
                self.width = int(val)
                self.height = int(val)
            self._camera_res = (
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_WIDTH), default=-1),
                as_int(self._get_prop(cv2.CAP_PROP_FRAME_HEIGHT), default=-1),
            )
            self._log(logging.INFO, f"CV2:Resolution:{self._camera_res[0]}x{self._camera_res[1]}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Resolution, camera not open!")

    @property
    def exposure(self):
        if self.cam_open:
            return self._get_prop(cv2.CAP_PROP_EXPOSURE)
        return float("nan")

    @exposure.setter
    def exposure(self, val) -> None:
        if val is None:
            self._log(logging.WARNING, f"CV2:Skipping set Exposure to {val}")
            return
        if self.cam_open:
            if isinstance(val, (int, float)) and val > 0:
                self._try_set_autoexposure("manual")
            if self._set_prop(cv2.CAP_PROP_EXPOSURE, val):
                self._log(logging.INFO, f"CV2:Exposure set:{val}")
                self._exposure = self._get_prop(cv2.CAP_PROP_EXPOSURE)
                self._log(logging.INFO, f"CV2:Exposure is:{self._exposure}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set Expsosure to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Exposure, camera not open!")

    @property
    def autoexposure(self):
        if self.cam_open:
            return self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
        return -1

    @autoexposure.setter
    def autoexposure(self, val) -> None:
        if val is None:
            self._log(logging.WARNING, f"CV2:Skipping set Autoexposure to:{val}")
            return
        if val == -1:
            return
        if self.cam_open:
            if val in (0, 1, 0.0, 1.0, True, False):
                mode = "auto" if bool(val) else "manual"
                ok = self._try_set_autoexposure(mode)
                self._autoexposure = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
                if not ok:
                    self._log(
                        logging.WARNING,
                        f"CV2:Autoexposure semantic set({mode}) may not be supported; readback={self._autoexposure}",
                    )
                return

            if self._set_prop(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                self._log(logging.INFO, f"CV2:Autoexposure set:{val}")
                self._autoexposure = self._get_prop(cv2.CAP_PROP_AUTO_EXPOSURE)
                self._log(logging.INFO, f"CV2:Autoexposure is:{self._autoexposure}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set Autoexposure to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Autoexposure, camera not open!")

    @property
    def fps(self):
        if self.cam_open:
            return self._get_prop(cv2.CAP_PROP_FPS)
        return float("nan")

    @fps.setter
    def fps(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"CV2:Skipping set FPS to:{val}")
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_FPS, val):
                self._log(logging.INFO, f"CV2:FPS set:{val}")
                self._framerate = self._get_prop(cv2.CAP_PROP_FPS)
                self._log(logging.INFO, f"CV2:FPS is:{self._framerate}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set FPS to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set FPS, camera not open!")

    @staticmethod
    def decode_fourcc(val):
        return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def fourcc(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_FOURCC), default=-1)
        return "None"

    @fourcc.setter
    def fourcc(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"CV2:Skipping set FOURCC to:{val}")
            return
        if self.cam_open:
            if isinstance(val, str):
                if self._set_prop(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(val[0], val[1], val[2], val[3])):
                    self._fourcc = self._get_prop(cv2.CAP_PROP_FOURCC)
                    self._fourcc_str = self.decode_fourcc(self._fourcc)
                    self._log(logging.INFO, f"CV2:FOURCC is:{self._fourcc_str}")
                else:
                    self._log(logging.ERROR, f"CV2:Failed to set FOURCC to:{val}")
            else:
                if self._set_prop(cv2.CAP_PROP_FOURCC, val):
                    self._fourcc = as_int(self._get_prop(cv2.CAP_PROP_FOURCC), default=-1)
                    self._fourcc_str = self.decode_fourcc(self._fourcc)
                    self._log(logging.INFO, f"CV2:FOURCC is:{self._fourcc_str}")
                else:
                    self._log(logging.ERROR, f"CV2:Failed to set FOURCC to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set fourcc, camera not open!")

    @property
    def buffersize(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_BUFFERSIZE), default=-1)
        return float("nan")

    @buffersize.setter
    def buffersize(self, val) -> None:
        if val is None or val < 0:
            self._log(logging.WARNING, f"CV2:Skipping set Buffersize to:{val}")
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_BUFFERSIZE, val):
                self._log(logging.INFO, f"CV2:Buffersize set:{val}")
                self._buffersize = as_int(self._get_prop(cv2.CAP_PROP_BUFFERSIZE), default=-1)
                self._log(logging.INFO, f"CV2:Buffersize is:{self._buffersize}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set Buffersize to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Buffersize, camera not open!")

    @property
    def gain(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_GAIN), default=-1)
        return float("nan")

    @gain.setter
    def gain(self, val) -> None:
        if val is None or val < 0:
            self._log(logging.WARNING, f"CV2:Skipping set Gain to:{val}")
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_GAIN, val):
                self._log(logging.INFO, f"CV2:Gain set:{val}")
                self._gain = as_int(self._get_prop(cv2.CAP_PROP_GAIN), default=-1)
                self._log(logging.INFO, f"CV2:Gain is:{self._gain}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set Gain to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set Gain, camera not open!")

    @property
    def wbtemperature(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_WB_TEMPERATURE), default=-1)
        return float("nan")

    @wbtemperature.setter
    def wbtemperature(self, val) -> None:
        if val is None or val < 0:
            self._log(logging.WARNING, f"CV2:Skipping set WB_TEMPERATURE to:{val}")
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_WB_TEMPERATURE, val):
                self._log(logging.INFO, f"CV2:WB_TEMPERATURE set:{val}")
                self._wbtemp = as_int(self._get_prop(cv2.CAP_PROP_WB_TEMPERATURE), default=-1)
                self._log(logging.INFO, f"CV2:WB_TEMPERATURE is:{self._wbtemp}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set whitebalance temperature to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set whitebalance temperature, camera not open!")

    @property
    def autowb(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_AUTO_WB), default=-1)
        return float("nan")

    @autowb.setter
    def autowb(self, val) -> None:
        if val is None or val < 0:
            self._log(logging.WARNING, f"CV2:Skipping set AUTO_WB to:{val}")
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_AUTO_WB, val):
                self._log(logging.INFO, f"CV2:AUTO_WB:{val}")
                self._autowb = as_int(self._get_prop(cv2.CAP_PROP_AUTO_WB), default=-1)
                self._log(logging.INFO, f"CV2:AUTO_WB is:{self._autowb}")
            else:
                self._log(logging.ERROR, f"CV2:Failed to set auto whitebalance to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set auto whitebalance, camera not open!")

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
            return float("nan")
        with self.cam_lock:
            try:
                return self.cam.get(prop_id)
            except Exception:
                return float("nan")

__all__ = ["cv2Core", "as_int", "as_float"]
