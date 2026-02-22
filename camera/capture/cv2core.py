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
# 2022 Added access to more opencv camera properties, auto populates missing configs
# 2021 Initializitation updates and use of queue
# 2019 Initial release, based on Bitbuckets FRC 4183 code
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: cv2Core
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
# - buffer: FrameBuffer    (allocated frame buffer, or None if closed)
# - metadata: dict         (latest frame metadata)
# - width / height / resolution / size
# - exposure, gain, autoexposure, autowb
# - fps, fourcc, buffer_count, flip
#
# Supported config parameters (configs dict):
# -------------------------------------------
#
# Core capture / output:
# - camera_res: tuple[int,int]      Capture size (w,h)
# - output_res: tuple[int,int]      Optional CPU resize target (applied in software)
# - fps: float|int                  Requested framerate
# - buffersize: int                 FrameBuffer capacity (default: 32)
# - buffer_overwrite: bool          If True, FrameBuffer overwrites oldest frames when full
#
# Formats:
# - format: str                     Desired raw stream format (e.g. 'YUYV', 'YUV420', 'SRGGB8')
# - fourcc: str|int                 Desired FOURCC (e.g. 'MJPG', 'YUY2', 'YU12')
# - convert_rgb: bool               If True (default), OpenCV backend converts to BGR.
#                                   If False, request raw and convert in software using 'format'.
# - convert_format: str             Target conversion format (default: 'BGR888')
#
# Controls (applied at open, and/or via properties/set_controls):
# - exposure: float                 Manual exposure (backend-dependent)
# - autoexposure: int               -1 leave unchanged, 0 AE off, 1 AE on
# - gain: float
# - wb_temp: int
# - autowb: int
#
# Low-level / advanced:
# - buffer_count: int|None          OpenCV CAP_PROP_BUFFERSIZE
# - flip: int                       0..7 (same enum as cv2Capture)
# - test_pattern: bool|str          Enable synthetic frame mode (for testing)
#
# Notes:
# - By default OpenCV delivers BGR frames; conversions run only when
#   convert_rgb=False or when convert_format requests RGB.
###############################################################################

from __future__ import annotations

from threading import Lock
from queue import Queue  # logging
import logging
import math
import sys
import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .framebuffer import FrameBuffer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def decode_fourcc(val) -> str:
    try:
        return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return ""

def probeCameras(numcams: int = 10) -> list[dict]:
    '''
    Scans cameras and returns default fourcc, width and height
    '''
    # check for up to 10 cameras
    index = 0
    arr = []
    i = numcams
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            tmp = cap.get(cv2.CAP_PROP_FOURCC)
            fourcc = "".join([chr((int(tmp) >> 8 * i) & 0xFF) for i in range(4)])
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            arr.append({"index": index, "fourcc": fourcc, "width": width, "height": height})
        index += 1
        i -= 1
    return arr

def findCamera(numcams: int = 10, fourccSig = "\x16\x00\x00\x00", widthSig: int=640, heightSig: int=480) -> int:
    '''
    Identifies camera with given default fourcc, width and height
    '''
    camprops = probeCameras(numcams)
    score = 0
    camera_index = 0
    for i in range(len(camprops)):
        try: found_fourcc = 1 if camprops[i]['fourcc'] == fourccSig else 0            
        except: found_fourcc = 0
        try: found_width = 1  if camprops[i]['width']  == widthSig  else 0
        except: found_width =  0
        try: found_height = 1 if camprops[i]['height'] == heightSig else 0   
        except: found_height = 0
        tmp = found_fourcc+found_width+found_height
        if tmp > score:
            score = tmp
            camera_index = i
    return camera_index
    
# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------

class cv2Core:
    """
    Non-threaded OpenCV capture core.

    Public methods:
    - open_cam()
    - close_cam()
    - capture_array()
    - apply_exposure_settings()

    Properties:
    - width, height, resolution, size
    - exposure, autoexposure
    - fps, fourcc, buffer_count
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

        # Normalize configs
        try:
            base_res = res if res is not None else self._configs.get("camera_res", (640, 480))
            if isinstance(base_res, (list, tuple)) and len(base_res) >= 2:
                base_res = (int(base_res[0]), int(base_res[1]))
            else:
                base_res = (640, 480)
            self._configs.setdefault("camera_res", base_res)
        except Exception:
            self._configs.setdefault("camera_res", (640, 480))

        # Requested settings
        self._exposure = exposure if exposure is not None else self._configs.get("exposure", -1.0)
        self._camera_res = self._configs.get("camera_res", (640, 480))
        self._output_res = self._configs.get("output_res", (-1, -1))
        self._framerate = self._configs.get("fps", -1.0)
        self._flip_method = self._configs.get("flip", 0)
        if "buffer_count" in self._configs:
            self._buffer_count = self._configs.get("buffer_count", 1)
        else:
            self._buffer_count = self._configs.get("buffersize", 1)
        self._fourcc = self._configs.get("fourcc", -1)
        self._autoexposure = self._configs.get("autoexposure", -1)
        self._gain = self._configs.get("gain", -1.0)
        self._wbtemp = self._configs.get("wb_temp", -1)
        self._autowb = self._configs.get("autowb", -1)
        self._settings = self._configs.get("settings", -1)

        self._output_width = int(self._output_res[0])
        self._output_height = int(self._output_res[1])

        self._capture_width = int(self._camera_res[0])
        self._capture_height = int(self._camera_res[1])

        self._requested_format = str(self._configs.get("format", "")).upper()
        self._convert_rgb = bool(self._configs.get("convert_rgb", True))
        self._convert_format = str(self._configs.get("convert_format", "BGR888")).upper()

        # Frame Buffer management
        self._buffer_capacity = int(self._configs.get("buffersize", 32))
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        self._buffer: FrameBuffer | None = None

        # Synthetic/test pattern mode
        self._test_pattern = self._configs.get("test_pattern", False)
        self._test_frame: np.ndarray | None = None
        self._test_next_t: float = 0.0

        # Conversion settings
        self._format: str | None = None
        self._cv2_conversion_code = None
        self._needs_bit_depth_conversion = False
        self._bit_depth_shift = 0
        self._needs_cpu_resize = False
        self._needs_cpu_flip = False
        self._needs_drop_alpha = False
        self._is_yuv420 = False

        # Runtime
        self.cam = None
        self.cam_open = False
        self.cam_lock = Lock()
        self._metadata: dict | None = None

        # Stats
        self.frame_time = 0.0
        self.measured_fps = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str) -> None:
        if self.log is None:
            return
        try:
            if not self.log.full():
                self.log.put_nowait((level, message))
        except Exception:
            pass

    @staticmethod
    def _is_raw_format(fmt: str | None) -> bool:
        if not fmt:
            return False
        fmt_u = str(fmt).upper()
        return fmt_u.startswith(("SRGGB", "SBGGR", "SGBRG", "SGRBG"))

    def _map_fourcc_format(self, fmt: str | int | None) -> str:
        if fmt is None:
            return ""
        if isinstance(fmt, int):
            fmt_s = decode_fourcc(fmt)
        else:
            fmt_s = str(fmt)
        fmt_u = fmt_s.strip().upper()
        if not fmt_u:
            return ""

        # Common mappings (FOURCC -> normalized format)
        mapping = {
            "BGR3": "BGR888",
            "RGB3": "RGB888",
            "BGRA": "XBGR8888",
            "RGBA": "XRGB8888",
            "YUY2": "YUYV",
            "YUYV": "YUYV",
            "YV12": "YUV420",
            "YU12": "YUV420",
            "I420": "YUV420",
            "MJPG": "MJPEG",
            "MJPEG": "MJPEG",
            "GRAY": "GRAY8",
            "GREY": "GRAY8",
        }
        return mapping.get(fmt_u, fmt_u)

    def _normalize_format(self, fmt: str | None, fourcc: str | int | None) -> str:
        mapped = self._map_fourcc_format(fmt) if fmt else ""
        if mapped:
            return mapped
        if fourcc not in (None, "", -1):
            return self._map_fourcc_format(fourcc)
        return ""

    def _set_color_format(self, fmt: str | None) -> None:
        try:
            fmt_upper = (fmt or "").upper()
        except Exception:
            fmt_upper = str(fmt).upper() if fmt is not None else ""

        self._format = fmt_upper if fmt_upper else None
        self._is_yuv420 = fmt_upper == "YUV420"
        self._needs_drop_alpha = fmt_upper in ("XRGB8888", "XBGR8888")
        self._update_conversion_settings()

    def _check_bit_depth_conversion(self, src_fmt: str, to_fmt: str) -> bool:
        if to_fmt not in ("BGR888", "RGB888"):
            return False
        bit_depth = 16
        for cand in (16, 14, 12, 10):
            if str(cand) in src_fmt:
                bit_depth = cand
                break
        self._bit_depth_shift = max(0, bit_depth - 8)
        return self._bit_depth_shift > 0

    def _update_conversion_settings(self) -> None:
        fmt = (self._format or "").upper()
        to = (self._convert_format or "BGR888").upper()

        self._cv2_conversion_code = None
        self._needs_bit_depth_conversion = False
        self._bit_depth_shift = 0

        if not fmt:
            return

        if (fmt == "BGR888" and to == "BGR888") or (fmt == "RGB888" and to == "RGB888"):
            return

        if fmt == "BGR888" and to == "RGB888":
            self._cv2_conversion_code = cv2.COLOR_BGR2RGB
            return
        if fmt == "RGB888" and to == "BGR888":
            self._cv2_conversion_code = cv2.COLOR_RGB2BGR
            return

        if fmt == "XBGR8888":
            self._cv2_conversion_code = cv2.COLOR_RGBA2BGR if to == "BGR888" else cv2.COLOR_RGBA2RGB
            return
        if fmt == "XRGB8888":
            self._cv2_conversion_code = cv2.COLOR_BGRA2BGR if to == "BGR888" else cv2.COLOR_BGRA2RGB
            return

        if fmt == "YUV420":
            self._cv2_conversion_code = cv2.COLOR_YUV2BGR_I420 if to == "BGR888" else cv2.COLOR_YUV2RGB_I420
            return
        if fmt == "YUYV":
            self._cv2_conversion_code = cv2.COLOR_YUV2BGR_YUY2 if to == "BGR888" else cv2.COLOR_YUV2RGB_YUY2
            return

        if fmt.startswith("SRGGB"):
            self._cv2_conversion_code = cv2.COLOR_BAYER_RG2BGR if to == "BGR888" else cv2.COLOR_BAYER_RG2RGB
            self._needs_bit_depth_conversion = self._check_bit_depth_conversion(fmt, to)
            return
        if fmt.startswith("SBGGR"):
            self._cv2_conversion_code = cv2.COLOR_BAYER_BG2BGR if to == "BGR888" else cv2.COLOR_BAYER_BG2RGB
            self._needs_bit_depth_conversion = self._check_bit_depth_conversion(fmt, to)
            return
        if fmt.startswith("SGBRG"):
            self._cv2_conversion_code = cv2.COLOR_BAYER_GB2BGR if to == "BGR888" else cv2.COLOR_BAYER_GB2RGB
            self._needs_bit_depth_conversion = self._check_bit_depth_conversion(fmt, to)
            return
        if fmt.startswith("SGRBG"):
            self._cv2_conversion_code = cv2.COLOR_BAYER_GR2BGR if to == "BGR888" else cv2.COLOR_BAYER_GR2RGB
            self._needs_bit_depth_conversion = self._check_bit_depth_conversion(fmt, to)
            return

    def _effective_output_size(self) -> tuple[int, int]:
        if self._output_width > 0 and self._output_height > 0:
            w, h = self._output_width, self._output_height
        else:
            w, h = self._capture_width, self._capture_height
        if self._flip_method in (1, 3, 5, 7):
            return (h, w)
        return (w, h)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_array(self) -> tuple[np.ndarray | None, float | None]:
        """Capture a single frame from the configured stream."""

        # Synthetic path
        if self._test_pattern:
            try:
                with self.cam_lock:
                    img = self._synthetic_frame()
                ts_ms = time.perf_counter() * 1000.0
                self._metadata = {"Timestamp": ts_ms, "Synthetic": True}
                return img, ts_ms
            except Exception as exc:
                self._log(logging.WARNING, f"CV2:Synthetic capture failed: {exc}")
                return None, None

        if not self.cam_open or self.cam is None:
            return None, None

        try:
            with self.cam_lock:
                ret, img = self.cam.read()
            if (not ret) or (img is None):
                return None, None

            ts_ms = time.perf_counter() * 1000.0

            # Apply conversion if needed
            code = self._cv2_conversion_code
            if code is not None:
                img = cv2.cvtColor(img, code)
                if self._needs_bit_depth_conversion:
                    shift = self._bit_depth_shift
                    if shift > 0:
                        img = (img >> shift).astype(np.uint8)

            if self._needs_drop_alpha and img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]

            if self._needs_cpu_resize:
                img = cv2.resize(img, (self._output_width, self._output_height))

            if self._needs_cpu_flip:
                flip = self._flip_method
                if flip == 1:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif flip == 2:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif flip == 3:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif flip == 4:
                    img = cv2.flip(img, 1)
                elif flip == 5:
                    img = cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                elif flip == 6:
                    img = cv2.flip(img, 0)
                elif flip == 7:
                    img = cv2.transpose(img)

            self._metadata = {"Timestamp": ts_ms}
            return img, ts_ms

        except Exception as exc:
            self._log(logging.WARNING, f"CV2:Capture failed: {exc}")
            return None, None

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open_cam(self) -> bool:
        """Open the camera and apply configuration properties."""

        # Always clear buffer before (re)opening
        self._buffer = None

        if self._test_pattern:
            try:
                self.cam = None
                self.cam_open = True
                self._set_color_format("BGR888")
                self._needs_cpu_resize = False
                self._needs_cpu_flip = bool(self._flip_method != 0)

                w, h = self._effective_output_size()
                self._buffer = FrameBuffer(
                    capacity=self._buffer_capacity,
                    frame_shape=(h, w, 3),
                    dtype=np.uint8,
                    overwrite=self._buffer_overwrite,
                )
                self._log(logging.INFO, f"CV2:Synthetic mode enabled pattern={self._test_pattern}")
                return True
            except Exception as exc:
                self._log(logging.CRITICAL, f"CV2:Failed to open synthetic camera: {exc}")
                self.cam = None
                self.cam_open = False
                return False

        try:
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
                return False

            # Apply settings to camera
            self.resolution = self._camera_res
            self.fps = self._framerate
            self.buffer_count = self._buffer_count
            self.fourcc = self._fourcc
            self.gain = self._gain
            self.wbtemperature = self._wbtemp
            self.autowb = self._autowb

            # Exposure settings are backend-dependent
            self.apply_exposure_settings()

            if self._settings > -1:
                ok = self._set_prop(cv2.CAP_PROP_SETTINGS, 0.0)
                if not ok:
                    self._log(logging.WARNING, "CV2:CAP_PROP_SETTINGS not supported by this backend/OS")

            # Request raw frames if desired
            if not self._convert_rgb:
                self._set_prop(cv2.CAP_PROP_CONVERT_RGB, 0.0)

            # Refresh actual settings
            self._camera_res = self.resolution
            self._capture_width, self._capture_height = self._camera_res
            self._framerate = self.fps
            self._autoexposure = self.autoexposure
            self._fourcc = self.fourcc
            self._fourcc_str = decode_fourcc(self._fourcc)
            self._gain = self.gain
            self._wbtemperature = self.wbtemperature
            self._autowb = self.autowb

            # Determine format + conversion strategy
            if self._convert_rgb:
                fmt = "BGR888"
            else:
                fmt = self._normalize_format(self._requested_format, self._fourcc)
                if not fmt:
                    fmt = "BGR888"

            self._set_color_format(fmt)

            # CPU resize/flip handling
            self._needs_cpu_resize = (self._output_width > 0 and self._output_height > 0)
            self._needs_cpu_flip = (self._flip_method != 0)

            # Allocate FrameBuffer for processed frames (BGR by default)
            w, h = self._effective_output_size()
            self._buffer = FrameBuffer(
                capacity=self._buffer_capacity,
                frame_shape=(h, w, 3),
                dtype=np.uint8,
                overwrite=self._buffer_overwrite,
            )

            self._log(
                logging.INFO,
                f"CV2:Open summary size={self._camera_res[0]}x{self._camera_res[1]} "
                f"format={self._format} fourcc={self._fourcc_str} "
                f"convert_rgb={self._convert_rgb} output={w}x{h}",
            )
            return True
        except Exception:
            self._log(logging.CRITICAL, "CV2:Exception during camera open!")
            self.cam = None
            self.cam_open = False
            return False

    def close_cam(self) -> None:
        """Release the underlying VideoCapture (idempotent)."""
        try:
            with self.cam_lock:
                cam = self.cam
                if cam is not None:
                    cam.release()
                self.cam = None
                self.cam_open = False
                self._buffer = None
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

        manual_requested = requested_exposure is not None and requested_exposure > 0
        if manual_requested:
            desired_ae_mode = "manual"
        else:
            if requested_ae is None or requested_ae == -1:
                desired_ae_mode = None
            else:
                desired_ae_mode = "auto" if requested_ae > 0 else "manual"

        if desired_ae_mode in ("auto", "manual"):
            self._try_set_autoexposure(desired_ae_mode)

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
    # Synthetic/test-pattern helpers
    # ------------------------------------------------------------------

    def _synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic frame matching the configured main-stream format."""
        try:
            fr = float(self._framerate or 0.0)
        except Exception:
            fr = 0.0

        if fr > 0.0:
            period = 1.0 / fr
            now = time.perf_counter()
            if self._test_next_t <= 0.0:
                self._test_next_t = now
            dt = self._test_next_t - now
            if dt > 0:
                time.sleep(dt)
            self._test_next_t = max(self._test_next_t + period, time.perf_counter())

        h = int(self._capture_height) if int(self._capture_height) > 0 else 480
        w = int(self._capture_width) if int(self._capture_width) > 0 else 640

        pat = self._test_pattern
        if pat is True:
            pat = "gradient"
        try:
            pat_s = str(pat).strip().lower()
        except Exception:
            pat_s = "gradient"

        if self._test_frame is not None and pat_s not in ("noise",):
            return self._test_frame

        if pat_s in ("static", "gradient"):
            x = np.linspace(0, 255, w, dtype=np.uint8)
            r = np.tile(x, (h, 1))
            g = np.tile(np.uint8(255) - x, (h, 1))
            b = np.full((h, w), 64, dtype=np.uint8)
            base_bgr = np.dstack((b, g, r))
        elif pat_s in ("checker", "checkerboard"):
            yy, xx = np.indices((h, w))
            block = 32
            chk = (((xx // block) + (yy // block)) & 1).astype(np.uint8) * 255
            base_bgr = np.dstack((chk, chk, chk))
        elif pat_s in ("noise", "random"):
            base_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            base_bgr = np.full((h, w, 3), 127, dtype=np.uint8)

        fmt = (self._format or "BGR888").upper()
        if fmt == "BGR888":
            frame = np.ascontiguousarray(base_bgr)
        elif fmt == "RGB888":
            frame = np.ascontiguousarray(base_bgr[:, :, ::-1])
        elif fmt == "XRGB8888":
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.ascontiguousarray(np.concatenate((base_bgr, alpha), axis=2))
        elif fmt == "XBGR8888":
            rgb = np.ascontiguousarray(base_bgr[:, :, ::-1])
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
        elif fmt == "YUV420":
            try:
                frame = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YUV_I420)
                frame = np.ascontiguousarray(frame)
            except Exception:
                frame = np.ascontiguousarray(base_bgr)
        elif fmt == "YUYV":
            try:
                frame = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YUV_YUY2)
                frame = np.ascontiguousarray(frame)
            except Exception:
                frame = np.ascontiguousarray(base_bgr)
        else:
            frame = np.ascontiguousarray(base_bgr)

        if pat_s not in ("noise",):
            self._test_frame = frame
        return frame

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
    def buffer(self) -> FrameBuffer | None:
        return self._buffer

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

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
    def size(self) -> tuple[int, int]:
        return (int(self._capture_width), int(self._capture_height))

    @size.setter
    def size(self, value) -> None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            self._log(logging.ERROR, "size must be a (width, height) tuple")
            return
        try:
            width, height = int(value[0]), int(value[1])
        except (ValueError, TypeError) as exc:
            self._log(logging.ERROR, f"Invalid size values: {exc}")
            return
        if width <= 0 or height <= 0:
            self._log(logging.ERROR, "size must be positive")
            return

        self._capture_width = width
        self._capture_height = height
        self._camera_res = (width, height)

        if self.cam_open:
            self._log(logging.INFO, f"CV2:Reopening camera to apply new size {width}x{height}")
            self.close_cam()
            self.open_cam()

    # Camera flip/rotation
    @property
    def flip(self) -> int:
        try:
            return int(self._flip_method)
        except Exception:
            return 0

    @flip.setter
    def flip(self, value) -> None:
        if value is None:
            self._log(logging.ERROR, "flip value cannot be None")
            return
        try:
            f = int(value)
        except (ValueError, TypeError) as exc:
            self._log(logging.ERROR, f"Invalid flip value: {exc}")
            return
        if f < 0 or f > 7:
            self._log(logging.ERROR, "flip must be in range 0..7")
            return
        self._flip_method = f
        try:
            self._configs["flip"] = f
        except Exception:
            pass
        self._needs_cpu_flip = (self._flip_method != 0)

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

    @property
    def fourcc(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_FOURCC), default=-1)
        return -1

    @fourcc.setter
    def fourcc(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"CV2:Skipping set FOURCC to:{val}")
            return
        if self.cam_open:
            if isinstance(val, str):
                if self._set_prop(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(val[0], val[1], val[2], val[3])):
                    self._fourcc = self._get_prop(cv2.CAP_PROP_FOURCC)
                    self._fourcc_str = decode_fourcc(self._fourcc)
                    self._log(logging.INFO, f"CV2:FOURCC is:{self._fourcc_str}")
                else:
                    self._log(logging.ERROR, f"CV2:Failed to set FOURCC to:{val}")
            else:
                if self._set_prop(cv2.CAP_PROP_FOURCC, val):
                    self._fourcc = as_int(self._get_prop(cv2.CAP_PROP_FOURCC), default=-1)
                    self._fourcc_str = decode_fourcc(self._fourcc)
                    self._log(logging.INFO, f"CV2:FOURCC is:{self._fourcc_str}")
                else:
                    self._log(logging.ERROR, f"CV2:Failed to set FOURCC to:{val}")
        else:
            self._log(logging.CRITICAL, "CV2:Failed to set fourcc, camera not open!")

    @property
    def buffer_count(self):
        if self.cam_open:
            return as_int(self._get_prop(cv2.CAP_PROP_BUFFERSIZE), default=-1)
        return float("nan")

    @buffer_count.setter
    def buffer_count(self, val) -> None:
        if val is None or val < 0:
            self._log(logging.WARNING, f"CV2:Skipping set Buffersize to:{val}")
            return
        if self.cam_open:
            if self._set_prop(cv2.CAP_PROP_BUFFERSIZE, val):
                self._log(logging.INFO, f"CV2:Buffersize set:{val}")
                self._buffer_count = as_int(self._get_prop(cv2.CAP_PROP_BUFFERSIZE), default=-1)
                self._log(logging.INFO, f"CV2:Buffersize is:{self._buffer_count}")
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

    # ------------------------------------------------------------------
    # Controls & metadata access
    # ------------------------------------------------------------------

    def get_control(self, name: str):
        if not name:
            return None
        key = str(name).strip()
        mapping = {
            "ExposureTime": self.exposure,
            "Exposure": self.exposure,
            "FrameRate": self.fps,
            "FPS": self.fps,
            "Gain": self.gain,
            "AutoExposure": self.autoexposure,
            "AeEnable": self.autoexposure,
            "AutoWB": self.autowb,
            "WBTemperature": self.wbtemperature,
            "FourCC": self.fourcc,
            "Width": self.width,
            "Height": self.height,
            "Resolution": self.resolution,
            "Timestamp": (self._metadata or {}).get("Timestamp"),
        }
        return mapping.get(key, None)

    def set_controls(self, controls: dict) -> bool:
        if not isinstance(controls, dict):
            return False
        ok_any = False
        for key, val in controls.items():
            k = str(key).strip()
            try:
                if k in ("ExposureTime", "Exposure"):
                    self.exposure = val
                elif k in ("FrameRate", "FPS"):
                    self.fps = val
                elif k in ("Gain",):
                    self.gain = val
                elif k in ("AutoExposure", "AeEnable"):
                    self.autoexposure = val
                elif k in ("AutoWB",):
                    self.autowb = val
                elif k in ("WBTemperature",):
                    self.wbtemperature = val
                elif k in ("FourCC",):
                    self.fourcc = val
                elif k in ("Resolution", "Size"):
                    self.size = val
                elif k in ("Flip",):
                    self.flip = val
                else:
                    continue
                ok_any = True
            except Exception:
                continue
        return ok_any

    def get_supported_main_color_formats(self) -> list[str]:
        formats = set()
        formats.add("BGR888")
        if self._convert_format:
            formats.add(self._convert_format)
        if self._requested_format:
            formats.add(self._map_fourcc_format(self._requested_format))
        if self._fourcc not in (-1, None, ""):
            formats.add(self._map_fourcc_format(self._fourcc))
        return sorted(f for f in formats if f)

    def get_supported_raw_color_formats(self) -> list[str]:
        formats = []
        for fmt in self.get_supported_main_color_formats():
            if self._is_raw_format(fmt):
                formats.append(fmt)
        return formats

    def get_supported_main_options(self):
        return [
            {
                "size": (self._capture_width, self._capture_height),
                "format": self._format or "BGR888",
                "fps": self._framerate,
            }
        ]

    def get_supported_raw_options(self):
        if self._is_raw_format(self._format):
            return [
                {
                    "size": (self._capture_width, self._capture_height),
                    "format": self._format,
                    "fps": self._framerate,
                }
            ]
        return []

    def log_stream_options(self) -> None:
        try:
            supported = self.get_supported_main_color_formats()
            self._log(logging.INFO, f"CV2:Configured format={self._format}, supported={supported}")
        except Exception:
            pass
        try:
            main_opts = self.get_supported_main_options()
            self._log(logging.INFO, f"CV2:Main options={main_opts}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Backend property helpers
    # ------------------------------------------------------------------

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


__all__ = ["cv2Core", "as_int", "as_float", "decode_fourcc"]
