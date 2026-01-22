###############################################################################
# Raspberry Pi CSI camera core (Picamera2 / libcamera)
#
# Shared core used by both non-Qt threaded capture and Qt wrappers.
#
# Urs Utzinger
# GPT-5.2 PCreation, Implementation, Testing, Documentation
# Sonnet 4.5 Performance Optimizations
###############################################################################

################################################################################
# Public API & Supported Config
#
# Class: PiCamera2Core
#
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
# - stream_policy: str              'default'|'maximize_fps_no_crop'|'maximize_fps_with_crop'|'maximize_fov'
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
from queue import Queue # logging
import logging
import time
import cv2
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .framebuffer import FrameBuffer

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None  # type: ignore

class PiCamera2Core:
    """Core Picamera2 wrapper.

    Responsibilities:
    - Normalize config
    - Open/configure/close Picamera2
    - Capture frames (raw or main)
    - Provide conversion and post-processing helpers
    - Provide metadata/control helpers

    Delivery mechanisms (Queue, Qt signals, etc.) are handled by wrappers.
    """

    # AeMeteringMode for libcamera / Picamera2 :
    #   0 = CentreWeighted, 1 = Spot, 2 = Matrix
    _AE_METERING_MAP = {
        "centre": 0,
        "center": 0,
        "centreweighted": 0,
        "centerweighted": 0,
        "average": 0,
        "spot": 1,
        "matrix": 2,
        "evaluative": 2,
    }

    # AwbMode for libcamera / Picamera2 :
    #   0 = Auto, 1 = Tungsten, 2 = Fluorescent,
    #   3 = Indoor, 4 = Daylight, 5 = Cloudy
    _AWB_MODE_MAP = {
        "auto": 0,
        "tungsten": 1,
        "incandescent": 1,
        "fluorescent": 2,
        "indoor": 3,
        "warm": 3,
        "daylight": 4,
        "sunny": 4,
        "cloudy": 5,
    }

    def __init__(
        self,
        configs: dict,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        log_queue: Queue | None = None,
    ):

        self._configs = configs or {}
        self.camera_num = int(camera_num)

        # Optional log sink compatible with existing capture modules
        self.log = log_queue

        # Thread-safety for Picamera2 access
        self.cam_lock = Lock()

        # Normalize configs: 
        #   derive main/raw keys from camera_res for consistent handling
        try:
            mode = str(self._configs.get("mode", "main")).lower()

            base_res = None
            if res is not None:
                base_res = res
            else:
                base_res = self._configs.get("camera_res", (640, 480))
            if isinstance(base_res, (list, tuple)) and len(base_res) >= 2:
                base_res = (int(base_res[0]), int(base_res[1]))
            else:
                base_res = (640, 480)
            self._configs.setdefault("camera_res", base_res)

            self._configs.setdefault(
                "main_format",
                str(self._configs.get("main_format", self._configs.get("format", "BGR3"))),
            )
            self._configs.setdefault(
                "raw_format",
                str(self._configs.get("raw_format", self._configs.get("format", "SRGGB8"))),
            )
            self._configs.setdefault("mode", mode)

        except (ValueError, TypeError) as exc:
            self._log(logging.ERROR, f"Config normalization error: {exc}")

        # Exposure override
        if exposure is not None:
            self._exposure = exposure
        else:
            self._exposure = self._configs.get("exposure", -1)

        # Resolution
        if res is not None:
            self._camera_res = tuple(res)
            self._configs["camera_res"] = tuple(res)
        else:
            self._camera_res = self._configs.get("camera_res", (640, 480))

        self._capture_width = int(self._camera_res[0])
        self._capture_height = int(self._camera_res[1])

        # Output (scaling)
        self._output_res = self._configs.get("output_res", (-1, -1))
        self._output_width = int(self._output_res[0])
        self._output_height = int(self._output_res[1])

        # Camera Controls
        self._framerate = self._configs.get("fps", 30)
        self._flip_method = self._configs.get("flip", 0)
        self._autoexposure = self._configs.get("autoexposure", -1)
        self._autowb = self._configs.get("autowb", -1)
        self._requested_format = str(self._configs.get("format", ""))
        self._fourcc = str(self._configs.get("fourcc", ""))
        self._mode = str(self._configs.get("mode", "main")).lower()
        policy_raw = str(self._configs.get("stream_policy", "default")).lower()
        if policy_raw in ("default", "maximize_fps_no_crop", "max_fps_no_crop", "maximize_fps_nocrop"):
            self._stream_policy = "maximize_fps_no_crop"
        elif policy_raw in ("maximize_fps_with_crop", "max_fps_with_crop", "maximize_fps_crop"):
            self._stream_policy = "maximize_fps_with_crop"
        elif policy_raw in ("maximize_fov", "max_fov"):
            self._stream_policy = "maximize_fov"
        else:
            self._stream_policy = "maximize_fps_no_crop"
            self._log(logging.WARNING, f"PiCam2:Unknown stream_policy '{policy_raw}', using default maximize_fps_no_crop")
        self._main_format = str(self._configs.get("main_format", self._requested_format))
        self._raw_format = str(self._configs.get("raw_format", self._requested_format))
        self._raw_res = self._configs.get("raw_res", (self._capture_width, self._capture_height))
        self._low_latency = bool(self._configs.get("low_latency", False))
        self._buffer_count = self._configs.get("buffer_count", None)
        self._hw_transform = bool(self._configs.get("hw_transform", 1))

        # Frame Buffer management
        self._buffer_capacity = int(self._configs.get("buffersize", 32))
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        # not used here self._buffer_copy_on_pull = bool(self._configs.get("buffer_copy", False))
        self._buffer = None

        # Synthetic/test pattern mode (bypasses Picamera2)
        # - False/None: normal camera
        # - True: enable with default pattern
        # - str: pattern name ('gradient', 'checker', 'noise', 'static')
        self._test_pattern = self._configs.get("test_pattern", False)
        self._test_frame: np.ndarray | None = None
        self._test_next_t: float = 0.0

        # Resolved state
        self._format: str | None = None
        self._stream_name: str = "main"
        self._needs_cpu_resize: bool = False
        self._needs_cpu_flip: bool = False
        self._is_yuv420: bool = False
        self._needs_drop_alpha: bool = False

        # Runtime
        self.picam2 = None
        self.cam_open = False
        self._metadata: dict | None = None
        self._last_metadata: dict | None = None
        self._last_set_controls: dict | None = None

        # Conversion caching
        self._convert_format = "BGR888"  # Default target format
        self._cv2_conversion_code = None
        self._needs_bit_depth_conversion = False
        self._bit_depth_shift = 0
        
        # ADD THESE:
        self._convert_buffer = None
        self._bitshift_buffer = None
        self._resize_buffer = None
        self._flip5_buffer = None
        
    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_array(self) -> tuple[np.ndarray | None, float | None]:
        """Capture a single frame from the configured stream.

        Returns (frame, ts_ms).

        - frame: numpy array, or None on failure.
        - ts_ms: best-effort timestamp in milliseconds derived from Picamera2
                metadata. The Picamera2 manual documents `SensorTimestamp` in
                nanoseconds since boot; this method converts it to milliseconds.
                This is a float to preserve sub-millisecond resolution.
        """

        # Synthetic path: generate frames without touching Picamera2
        # ----------------------------------------------------------------------------
        if self._test_pattern:
            try:
                with self.cam_lock:
                    img = self._synthetic_frame()
                ts_ms = time.perf_counter() * 1000.0
                self._metadata = {
                    "SensorTimestamp": None,
                    "Timestamp": ts_ms,
                    "FrameDuration": None,
                    "FrameDurationLimits": None,
                    "ScalerCrop": None,
                    "Synthetic": True,
                }
                return img, ts_ms
            except Exception as exc:
                self._log(logging.WARNING, f"PiCam2:Synthetic capture failed: {exc}")
                return None, None

        if self.picam2 is None:
            return None, None

        # Actual Picamera2 capture path
        # --------------------------------------------------------------------------

        try:
            # Capture frame and metadata
            with self.cam_lock:
                request = self.picam2.capture_request()            

            try:
                # Single call gets both array and metadata
                img = request.make_array(self._stream_name)
                md = request.get_metadata()
                
                # Extract timestamp - optimized path
                sensor_ts = md.get("SensorTimestamp")
                ts_ms = float(sensor_ts) / 1_000_000.0 if sensor_ts else time.perf_counter() * 1000.0
                
                self._metadata = md
                
            finally:
                # Always release the request
                request.release()
        
            # CPU processing (only if ISP couldn't handle it)
            # ------------------------------------------------
            
            # Color conversion (only for RAW or unsupported formats)
            code = self._cv2_conversion_code
            if code is not None:
                if self._convert_buffer is not None:
                    # Use pre-allocated buffer
                    cv2.cvtColor(img, code, dst=self._convert_buffer)
                    img = self._convert_buffer
                else:
                    img = cv2.cvtColor(img, code)
                
                # Bit depth conversion (RAW formats)
                if self._needs_bit_depth_conversion:
                    shift = self._bit_depth_shift
                    if shift > 0:
                        if self._bitshift_buffer is not None:
                            np.right_shift(img, shift, out=self._bitshift_buffer, casting='unsafe')
                            img = self._bitshift_buffer
                        else:
                            img = (img >> shift).astype(np.uint8)
            
            # Drop alpha channel if needed (XRGB/XBGR formats)
            if self._needs_drop_alpha:
                img = img[:, :, :3]

            # CPU resize (only if ISP couldn't scale)
            if self._needs_cpu_resize:
                if self._resize_buffer is not None:
                    cv2.resize(img, self._output_res, dst=self._resize_buffer)
                    img = self._resize_buffer
                else:
                    img = cv2.resize(img, self._output_res)
            
            # CPU flip/rotation (only if Transform wasn't applied)
            if self._needs_cpu_flip:
                flip = self._flip_method
                if flip == 1:  # ccw 90
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif flip == 2:  # rot 180
                    img = cv2.rotate(img, cv2.ROTATE_180)
                elif flip == 3:  # cw 90
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif flip == 4:  # horizontal
                    img = cv2.flip(img, 1)
                elif flip == 5:  # upright diagonal
                    if self._flip5_buffer is not None:
                        temp = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        cv2.flip(temp, 1, dst=self._flip5_buffer)
                        img = self._flip5_buffer
                    else:
                        img = cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                elif flip == 6:  # vertical
                    img = cv2.flip(img, 0)
                elif flip == 7:  # upperleft diagonal
                    img = cv2.transpose(img)
            
            return img, ts_ms

        except Exception as exc:
            self._log(logging.WARNING, f"PiCam2:Capture failed: {exc}")
            return None, None

    def _allocate_processing_buffers(self):
        """Allocate pre-sized buffers for CPU processing operations.
        
        Called in open_cam() after configuration is determined.
        Avoids per-frame allocations during capture.
        """
        h, w = self._capture_height, self._capture_width
        
        # Clear any existing buffers
        self._convert_buffer = None
        self._bitshift_buffer = None
        self._resize_buffer = None
        self._flip5_buffer = None
        
        # Conversion buffer (for color space conversions)
        if self._cv2_conversion_code is not None:
            if self._is_yuv420:
                # YUV420 is H*1.5 x W, converts to H x W x 3
                self._convert_buffer = np.empty((h, w, 3), dtype=np.uint8)
            elif self._format in ("YUYV",):
                # YUYV is H x W x 2, converts to H x W x 3
                self._convert_buffer = np.empty((h, w, 3), dtype=np.uint8)
            else:
                # Most formats: same spatial dimensions
                self._convert_buffer = np.empty((h, w, 3), dtype=np.uint8)
        
        # Bit shift buffer (for RAW formats with >8 bit depth)
        if self._needs_bit_depth_conversion:
            self._bitshift_buffer = np.empty((h, w, 3), dtype=np.uint8)
        
        # Resize buffer (when ISP can't scale or for RAW)
        if self._needs_cpu_resize:
            self._resize_buffer = np.empty(
                (self._output_height, self._output_width, 3), 
                dtype=np.uint8
            )
        
        # Flip mode 5 needs intermediate buffer
        if self._needs_cpu_flip and self._flip_method == 5:
            self._flip5_buffer = np.empty((w, h, 3), dtype=np.uint8)  # Note: transposed dimensions
            
    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open_cam(self) -> bool:
        """Open and configure the Picamera2 camera with optimized ISP usage.

        This method configures the requested stream (main/raw), uses ISP hardware
        for format conversion, scaling, and transforms when possible, and only
        falls back to CPU processing when necessary (e.g., RAW debayering).
        """

        # Always clear buffer before (re)opening
        self._buffer = None
        
        # Clear processing buffers
        self._convert_buffer = None
        self._bitshift_buffer = None
        self._resize_buffer = None
        self._flip5_buffer = None

        # Synthetic/test-pattern mode: open without Picamera2 installed.
        # ----------------------------------------------------------------------------
        if self._test_pattern:
            try:
                self.picam2 = None
                self.cam_open = True

                self._stream_name = "main"

                req = str(self._requested_format or self._fourcc or "").upper()
                pfmt = self._main_format or req
                mapped = self._map_fourcc_format(pfmt) or "BGR888"

                if self._is_raw_format(mapped):
                    mapped = "BGR888"
                if mapped not in self._supported_main_color_formats():
                    mapped = "BGR888"
                if mapped == "MJPEG":
                    mapped = "BGR888"

                self._set_color_format(mapped)
                self._needs_cpu_resize = False
                self._needs_cpu_flip = bool(self._flip_method != 0)

                # Reset pacing state
                self._test_next_t = 0.0
                self._test_frame = None

                # Allocate frame buffer for synthetic mode (always 3 channel 8-bit)
                self._buffer = FrameBuffer(
                    capacity    = self._buffer_capacity,
                    frame_shape = (self._capture_height, self._capture_width, 3),
                    dtype       = np.uint8,
                    overwrite   = self._buffer_overwrite
                )

                self._log(logging.INFO, f"PiCam2:Synthetic mode enabled pattern={self._test_pattern}")
                self._log(
                    logging.INFO,
                    f"PiCam2:Open summary stream=main size={self._capture_width}x{self._capture_height} fmt={self._format} req_fps={self._framerate} Synthetic=True",
                )
                return True
            except Exception as exc:
                self.picam2 = None
                self.cam_open = False
                self._log(logging.CRITICAL, f"PiCam2:Failed to open synthetic camera: {exc}")
                return False

        # Make sure Picamera2 is installed
        # ----------------------------------------------------------------------------
        if Picamera2 is None:
            self._log(logging.CRITICAL, "PiCam2:picamera2 is not installed")
            self.cam_open = False
            return False

        # Actual Picamera2 open path
        # ----------------------------------------------------------------------------
        try:
            self.picam2 = Picamera2(camera_num=self.camera_num)

            req = str(self._requested_format or self._fourcc or "").upper()

            # RAW stream selection ---------------------------------------
            if self._mode == "raw":
                self._stream_name = "raw"

                # Determine requested raw format
                if self._is_raw_format(self._raw_format):
                    rf = self._raw_format
                elif self._is_raw_format(req):
                    rf = req
                else:
                    rf = "SRGGB8"  # Default fallback
                    self._log(logging.INFO, f"PiCam2:No valid RAW format specified; defaulting to {rf}")
        
                try:
                    raw_w, raw_h = int(self._raw_res[0]), int(self._raw_res[1])
                    raw_size = (raw_w, raw_h)
                except Exception:
                    raw_size = (self._capture_width, self._capture_height)

                orig_rf, orig_raw_size = rf, raw_size
                rf, raw_size = self._validate_raw_selection(rf, raw_size)

                selected_raw = self._select_raw_sensor_mode(desired_size=raw_size)
                if selected_raw is not None:
                    raw_size = selected_raw["size"]
                    rf = selected_raw["format"]
                    try:
                        self._capture_width, self._capture_height = int(raw_size[0]), int(raw_size[1])
                    except Exception:
                        pass
                    req_str = (
                        f"{orig_rf}@{orig_raw_size[0]}x{orig_raw_size[1]}" if isinstance(orig_raw_size, tuple) else str(orig_rf)
                    )
                    sel_str = f"{rf}@{raw_size[0]}x{raw_size[1]}"
                    self._log(
                        logging.INFO,
                        f"PiCam2:RAW sensor selection policy={self._stream_policy} requested={req_str} selected={sel_str} fps~{selected_raw['fps']}",
                    )
                picam_format = rf
                self._set_color_format(picam_format)

            # Main stream selection ---------------------------------------
            else:
                self._stream_name = "main"
                
                # Determine target format - prefer convert_format if set, otherwise main_format
                target_format = self._convert_format or self._main_format or req
                mapped = self._map_fourcc_format(target_format) or "BGR888"
                
                if mapped not in self.get_supported_main_color_formats():
                    self._log(
                        logging.INFO,
                        f"PiCam2:Main format '{mapped}' is not in common formats list; falling back to BGR888."
                    )
                    mapped = "BGR888"
                
                picam_format = mapped
                self._set_color_format(picam_format)

            # Determine target size for main stream
            # ----------------------------------------------------------------------------
            if self._stream_name == "main":
                # For main stream, use output_res if specified, otherwise camera_res
                if (self._output_width > 0) and (self._output_height > 0):
                    target_size = (self._output_width, self._output_height)
                else:
                    target_size = (self._capture_width, self._capture_height)
            else:
                # For raw stream, always use the sensor mode size
                target_size = (self._capture_width, self._capture_height)

            # Set up transform if requested
            # ----------------------------------------------------------------------------
            transform = None
            transform_applied = False
            if self._hw_transform and (self._flip_method != 0):
                try:
                    from libcamera import Transform  # type: ignore
                    transform = self._flip_to_transform(self._flip_method, Transform)
                    if transform is not None:
                        transform_applied = True
                except Exception:
                    transform = None

            # Prepare frame duration controls
            # ----------------------------------------------------------------------------
            try:
                fr = float(self._framerate or 0.0)
            except Exception:
                fr = 0.0

            cfg_controls: dict[str, object] = {}
            if fr > 0.0:
                try:
                    frame_us = int(round(1_000_000.0 / fr))
                    if frame_us > 0:
                        cfg_controls["FrameDurationLimits"] = (frame_us, frame_us)
                except Exception:
                    cfg_controls = {"FrameRate": fr}

            # Sensor mode selection for main stream
            # ----------------------------------------------------------------------------
            sensor_cfg = None
            if self._stream_name == "main":
                try:
                    selected_main_sensor = self._select_main_sensor_mode(desired_size=target_size)
                    if selected_main_sensor is not None:
                        sensor_cfg = {"output_size": tuple(selected_main_sensor["size"])}
                        
                        bd = selected_main_sensor.get("bit_depth")
                        if bd is not None:
                            try:
                                sensor_cfg["bit_depth"] = int(bd)
                            except Exception:
                                pass
                        
                        fps_est = selected_main_sensor.get("fps")
                        self._log(
                            logging.INFO,
                            f"PiCam2:MAIN sensor selection policy={self._stream_policy} "
                            f"desired_main={target_size[0]}x{target_size[1]} "
                            f"selected_sensor={sensor_cfg.get('output_size')} bit_depth={sensor_cfg.get('bit_depth', 'n/a')} "
                            f"fps~{fps_est}",
                        )
                    else:
                        self._log(
                            logging.INFO,
                            f"PiCam2:MAIN sensor selection policy={self._stream_policy} "
                            f"desired_main={target_size[0]}x{target_size[1]} selected_sensor=None (libcamera will choose)",
                        )
                except Exception:
                    sensor_cfg = None

            # Buffer count
            # ----------------------------------------------------------------------------
            buffer_count = self._buffer_count
            if buffer_count is None and self._low_latency:
                buffer_count = 3
            
            # Create configuration
            # ----------------------------------------------------------------------------
            config = None
            
            # TRY 1: Optimal configuration - let ISP do all the work
            # ----------------------------------------------------------------------------
            if self._stream_name == "main":
                try:
                    picam_kwargs = {
                        "main": {"size": target_size, "format": picam_format},
                        "controls": cfg_controls
                    }
                    
                    if buffer_count is not None:
                        picam_kwargs["buffer_count"] = int(buffer_count)
                    
                    if transform is not None:
                        picam_kwargs["transform"] = transform
                    
                    if sensor_cfg is not None:
                        picam_kwargs["sensor"] = sensor_cfg
                    
                    config = self.picam2.create_video_configuration(**picam_kwargs)
                    
                    # SUCCESS: ISP will handle format, scaling, and transform!
                    self._needs_cpu_resize = False
                    self._needs_cpu_flip = False
                    self._cv2_conversion_code = None
                    self._needs_drop_alpha = False
                    self._needs_bit_depth_conversion = False
                    
                    # Update actual capture dimensions to match ISP output
                    self._capture_width = target_size[0]
                    self._capture_height = target_size[1]
                    
                    self._log(
                        logging.INFO,
                        f"PiCam2:ISP configuration successful - hardware will handle format={picam_format}, "
                        f"size={target_size}, transform={transform_applied}"
                    )
                    
                except Exception as exc:
                    self._log(logging.WARNING, f"PiCam2:ISP optimal config failed: {exc}, trying fallback")
                    config = None
            
            # RAW stream configuration
            # ----------------------------------------------------------------------------
            elif self._stream_name == "raw":
                try:
                    picam_kwargs = {
                        "raw": {"size": raw_size, "format": picam_format},
                        "controls": cfg_controls
                    }
                    
                    if buffer_count is not None:
                        picam_kwargs["buffer_count"] = int(buffer_count)
                    
                    if transform is not None:
                        picam_kwargs["transform"] = transform
                    
                    config = self.picam2.create_video_configuration(**picam_kwargs)
                    
                    # RAW always needs CPU processing for debayering
                    self._needs_cpu_resize = (self._output_width > 0 and self._output_height > 0)
                    self._needs_cpu_flip = (self._flip_method != 0) and (not transform_applied)
                    
                    # Update conversion settings for RAW
                    self._update_conversion_settings()
                    
                except Exception:
                    # Fallback for RAW
                    for fmt in ("SRGGB8", "SRGGB10_CSI2P"):
                        try:
                            picam_kwargs = {"raw": {"size": raw_size, "format": fmt}, "controls": cfg_controls}
                            if transform is not None:
                                picam_kwargs["transform"] = transform
                            config = self.picam2.create_video_configuration(**picam_kwargs)
                            self._set_color_format(fmt)
                            self._update_conversion_settings()
                            break
                        except Exception:
                            continue

            # FALLBACK 1: Try common main formats if initial config failed
            # ----------------------------------------------------------------------------
            if config is None and self._stream_name == "main":
                for fmt in ("BGR888", "RGB888"):
                    try:
                        picam_kwargs = {
                            "main": {"size": target_size, "format": fmt},
                            "controls": cfg_controls
                        }
                        if transform is not None:
                            picam_kwargs["transform"] = transform
                        if buffer_count is not None:
                            picam_kwargs["buffer_count"] = int(buffer_count)
                        
                        config = self.picam2.create_video_configuration(**picam_kwargs)
                        
                        self._set_color_format(fmt)
                        self._needs_cpu_resize = False
                        self._needs_cpu_flip = False
                        self._cv2_conversion_code = None
                        self._needs_drop_alpha = False
                        self._needs_bit_depth_conversion = False
                        
                        self._capture_width = target_size[0]
                        self._capture_height = target_size[1]
                        
                        self._log(logging.INFO, f"PiCam2:Fallback to {fmt} successful")
                        break
                    except Exception:
                        continue

            # FALLBACK 2: Drop controls entirely if platform rejects timing controls
            # ----------------------------------------------------------------------------
            if config is None:
                try:
                    if self._stream_name == "raw":
                        picam_kwargs = {"raw": {"size": raw_size, "format": picam_format}, "controls": {}}
                    else:
                        picam_kwargs = {"main": {"size": target_size, "format": picam_format}, "controls": {}}
                    
                    if transform is not None:
                        picam_kwargs["transform"] = transform
                    if buffer_count is not None:
                        picam_kwargs["buffer_count"] = int(buffer_count)
                    
                    config = self.picam2.create_video_configuration(**picam_kwargs)
                    
                    self._log(logging.WARNING, "PiCam2:Had to drop timing controls to create configuration")
                except Exception as exc:
                    self._log(logging.CRITICAL, f"PiCam2:All configuration attempts failed: {exc}")
                    self.picam2 = None
                    self.cam_open = False
                    return False

            # Apply configuration
            # ----------------------------------------------------------------------------
            self.picam2.configure(config)

            # Build camera controls
            # ----------------------------------------------------------------------------
            controls = {}

            # ScalerCrop for main stream (maximize_fov and maximize_fps_with_crop only)
            if self._stream_name == "main" and self._stream_policy in ("maximize_fov", "maximize_fps_with_crop"):
                try:
                    props = getattr(self.picam2, "camera_properties", {})
                    crop_rect = None
                    paa = None
                    if isinstance(props, dict):
                        paa = props.get("PixelArrayActiveAreas") or props.get("ActiveArea")
                    if paa:
                        rect = paa[0] if isinstance(paa, (list, tuple)) and len(paa) > 0 else paa
                        if isinstance(rect, (list, tuple)) and len(rect) == 4:
                            crop_rect = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
                    if crop_rect is None:
                        pas = props.get("PixelArraySize") if isinstance(props, dict) else None
                        if isinstance(pas, (list, tuple)) and len(pas) == 2:
                            crop_rect = (0, 0, int(pas[0]), int(pas[1]))
                    if crop_rect is not None:
                        controls["ScalerCrop"] = crop_rect
                except Exception:
                    pass

            # Exposure controls
            exposure  = self._exposure
            autoexp   = self._autoexposure
            autowb    = self._autowb

            manual_requested = exposure is not None and exposure > 0
            if manual_requested:
                controls["AeEnable"] = False
                controls["ExposureTime"] = int(exposure)
            else:
                if autoexp is not None and autoexp != -1:
                    controls["AeEnable"] = bool(autoexp)

            try:
                cfg_meter = self._configs.get("aemeteringmode", 0)
                meter_val = self._parse_aemeteringmode(cfg_meter)
                controls.setdefault("AeMeteringMode", meter_val)
            except Exception:
                pass

            # AWB controls
            if autowb is not None and autowb != -1:
                controls["AwbEnable"] = bool(autowb)

            try:
                cfg_awbmode = self._configs.get("awbmode", 0)
                awbmode_val = self._parse_awbmode(cfg_awbmode)
                controls.setdefault("AwbMode", awbmode_val)
            except Exception:
                pass

            # Start camera and apply controls
            # ----------------------------------------------------------------------------
            self.picam2.start()
            self.cam_open = True

            if controls:
                ok = self._set_controls(controls)
                if ok:
                    self._log(logging.INFO, f"PiCam2:Controls set {controls}")

            # Allocate processing buffers based on what CPU processing is needed
            # ----------------------------------------------------------------------------
            self._allocate_processing_buffers()

            # Log final configuration summary
            # ----------------------------------------------------------------------------
            try:
                md = None
                with self.cam_lock:
                    md = self._capture_metadata()
                if isinstance(md, dict):
                    fd = md.get("FrameDuration")
                    fdl = md.get("FrameDurationLimits")
                    sc = md.get("ScalerCrop")
                    # Use target_size for both, or check if raw_size exists
                    if self._stream_name == 'main':
                        actual_size = target_size
                    else:
                        actual_size = (self._capture_width, self._capture_height)  # or raw_size if defined
                    self._log(
                        logging.INFO,
                        f"PiCam2:Open summary stream={self._stream_name} size={actual_size} "
                        f"fmt={self._format} req_fps={self._framerate} "
                        f"FrameDuration={fd} FrameDurationLimits={fdl} ScalerCrop={sc} "
                        f"cpu_resize={self._needs_cpu_resize} cpu_flip={self._needs_cpu_flip} "
                        f"cpu_convert={self._cv2_conversion_code is not None}"
                    )
            except Exception:
                pass

            self._log(logging.INFO, "PiCam2:Camera opened")
            
            # Allocate frame buffer after stream is open and size is known
            self._buffer = FrameBuffer(
                capacity    = self._buffer_capacity,
                frame_shape = (self._capture_height, self._capture_width, 3),
                dtype       = np.uint8,
                overwrite   = self._buffer_overwrite
            )
            
            return True

        except Exception as exc:
            self.picam2 = None
            self.cam_open = False
            self._log(logging.CRITICAL, f"PiCam2:Failed to open camera: {exc}")
            return False

    def close_cam(self):
        """Stop and close the camera, releasing resources."""
        try:
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                except Exception as exc:
                    self._log(logging.ERROR, f"Error during camera stop: {exc}")
                try:
                    self.picam2.close()
                except Exception as exc:
                    self._log(logging.ERROR, f"Error during camera close: {exc}")
        finally:
            self.picam2 = None
            self._buffer = None
            self.cam_open = False
            self._last_metadata = None
            self._test_frame = None
            self._test_next_t = 0.0

    # ------------------------------------------------------------------
    # Synthetic/test-pattern helpers
    # ------------------------------------------------------------------

    def _synthetic_frame(self) -> np.ndarray:
        """Generate a synthetic frame matching the configured main-stream format.

        This bypasses Picamera2 and is intended for profiling the wrapper thread
        + FrameBuffer behavior without camera/libcamera overhead.
        """
        # Pace generation if a framerate was requested.
        try:
            fr = float(self._framerate or 0.0)
        except Exception:
            fr = 0.0

        if fr > 0.0:
            period = 1.0 / fr
            now = time.perf_counter()
            if self._test_next_t <= 0.0:
                self._test_next_t = now
            # Sleep until the next frame time.
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

        # Static frame cache for non-noise patterns.
        if self._test_frame is not None and pat_s not in ("noise",):
            return self._test_frame

        if pat_s in ("static", "gradient"):
            # Simple horizontal gradient in B,G,R.
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
            # Default: mid-gray
            base_bgr = np.full((h, w, 3), 127, dtype=np.uint8)

        fmt = (self._format or "BGR888").upper()

        if fmt == "BGR888":
            frame = np.ascontiguousarray(base_bgr)
        elif fmt == "RGB888":
            frame = np.ascontiguousarray(base_bgr[:, :, ::-1])
        elif fmt == "XRGB8888":
            # Per our convert() comment: XRGB8888 appears as BGRA in Python
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.ascontiguousarray(np.concatenate((base_bgr, alpha), axis=2))
        elif fmt == "XBGR8888":
            # Per our convert() comment: XBGR8888 appears as RGBA in Python
            rgb = np.ascontiguousarray(base_bgr[:, :, ::-1])
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            frame = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
        elif fmt == "YUV420":
            # Match cv2.COLOR_YUV2*I420 expectations: (H*3/2, W) uint8
            try:
                frame = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YUV_I420)
                frame = np.ascontiguousarray(frame)
            except Exception:
                frame = np.ascontiguousarray(base_bgr)
        elif fmt == "YUYV":
            # Match cv2.COLOR_YUV2*YUY2 expectations: (H, W, 2) uint8
            try:
                frame = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YUV_YUY2)
                frame = np.ascontiguousarray(frame)
            except Exception:
                frame = np.ascontiguousarray(base_bgr)
        else:
            # Unknown/unsupported synthetic format; fall back to BGR888.
            frame = np.ascontiguousarray(base_bgr)

        # Cache static patterns.
        if pat_s not in ("noise",):
            self._test_frame = frame
        return frame

    # ---------------------------------------------------------------------
    # Methods
    # ---------------------------------------------------------------------

    def get_supported_main_options(self):
        """Get supported main stream options for BGR888."""
        modes_sorted = []
        if self.picam2 is None:
            return modes_sorted

        modes = self._list_sensor_modes()
        if not modes:
            return modes_sorted

        def mode_area(m: dict) -> int:
            try:
                a = m.get("area")
                if a is not None:
                    return int(a)
                w, h = m.get("size", (0, 0))
                return int(w) * int(h)
            except Exception:
                return 0

        max_area = max(mode_area(m) for m in modes)
        full_fov_threshold = 0.95 * float(max_area)

        seen_sizes = set()

        if self._stream_policy in ("maximize_fps_no_crop", "maximize_fps_with_crop"):
            modes.sort(key=lambda m: (-float(m.get("fps", 0.0) or 0.0), -mode_area(m)))
        else:
            modes.sort(key=lambda m: (-mode_area(m), -float(m.get("fps", 0.0) or 0.0)))

        for m in modes:
            (w, h) = m["size"]
            if (w, h) in seen_sizes:
                continue
            seen_sizes.add((w, h))
            modes_sorted.append({
                "camera_res": (w, h),
                "output_res": (w, h),
                "main_format": "BGR888",
                "max_fps": float(m.get("fps", 0.0) or 0.0),
                "full_fov": bool(float(mode_area(m)) >= full_fov_threshold),
            })

        return modes_sorted

    def get_supported_main_color_formats(self):
        """Return common main-stream pixel formats (libcamera/Picamera2 naming).

        Note:
        - Actual availability depends on the camera and pipeline.
        - The Picamera2 manual highlights these as commonly used formats:
            XBGR8888, XRGB8888, RGB888, BGR888, YUV420.
        - YUYV/MJPEG are also mentioned for (non-CSI) cameras where supported.
        """
        return list(self._supported_main_color_formats())

    @staticmethod
    def _supported_main_color_formats() -> tuple[str, ...]:
        """Canonical list of main-stream formats used across helpers.

        Keep this in one place so FOURCC mapping, format validation, and other
        helpers stay consistent.
        """
        return (
            "XBGR8888",
            "XRGB8888",
            "RGB888",
            "BGR888",
            "YUV420",
            "YUYV",
            "MJPEG",
        )

    def get_supported_raw_options(self):
        """List all supported raw sensor modes from Picamera2."""
        return self._list_raw_sensor_modes()

    def get_supported_raw_color_formats(self) -> list[str]:
        """Return supported raw Bayer format strings.

        This is derived from the camera's advertised raw sensor modes.
        If the camera is not open (or modes are unavailable), returns an
        empty list.
        """
        fmts: set[str] = set()
        for m in self._list_raw_sensor_modes():
            try:
                f = m.get("format")
                fs = self._format_upper(f)
                if fs:
                    fmts.add(fs)
            except Exception:
                continue
        return sorted(fmts)

    def get_control(self, name: str):
        return self._get_control(name)

    def set_controls(self, controls: dict) -> bool:
        return self._set_controls(controls)

    def _set_color_format(self, fmt: str | None):
        """Set and analyze the requested format string."""
        try:
            fmt_upper = (fmt or "").upper()
        except Exception:
            fmt_upper = str(fmt).upper() if fmt is not None else ""

        self._format = fmt_upper if fmt_upper else None
        self._is_yuv420 = fmt_upper == "YUV420"
        non_alpha = set(self._supported_main_color_formats()) - {"XBGR8888", "XRGB8888"}
        self._needs_drop_alpha = bool(fmt_upper) and (not self._is_raw_format(fmt_upper)) and (fmt_upper not in non_alpha)
        
        if fmt_upper and self._needs_drop_alpha:
            self._log(logging.INFO, f"PiCam2:Dropping alpha channel for {fmt_upper} frames")
        
        # Pre-compute conversion settings
        self._update_conversion_settings()

    def _update_conversion_settings(self):
        """Pre-compute and cache color conversion settings.
        
        Called whenever _format or _convert_format changes.
        """
        fmt = (self._format or "").upper()
        to = (self._convert_format or "BGR888").upper()
        
        # Reset conversion state
        self._cv2_conversion_code = None
        self._needs_bit_depth_conversion = False
        self._bit_depth_shift = 0
        
        if not fmt:
            return
        
        # Check if no conversion needed
        if (fmt == "BGR888" and to == "BGR888") or (fmt == "RGB888" and to == "RGB888"):
            return
        
        # Simple BGR<->RGB swaps
        if fmt == "BGR888" and to == "RGB888":
            self._cv2_conversion_code = cv2.COLOR_BGR2RGB
            return
        if fmt == "RGB888" and to == "BGR888":
            self._cv2_conversion_code = cv2.COLOR_RGB2BGR
            return
        
        # 32-bit formats with alpha
        if fmt == "XBGR8888":
            self._cv2_conversion_code = cv2.COLOR_RGBA2BGR if to == "BGR888" else cv2.COLOR_RGBA2RGB
            return
        if fmt == "XRGB8888":
            self._cv2_conversion_code = cv2.COLOR_BGRA2BGR if to == "BGR888" else cv2.COLOR_BGRA2RGB
            return
        
        # YUV formats
        if fmt == "YUV420":
            self._cv2_conversion_code = cv2.COLOR_YUV2BGR_I420 if to == "BGR888" else cv2.COLOR_YUV2RGB_I420
            return
        if fmt == "YUYV":
            self._cv2_conversion_code = cv2.COLOR_YUV2BGR_YUY2 if to == "BGR888" else cv2.COLOR_YUV2RGB_YUY2
            return
        
        # Bayer/RAW formats
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

    def _check_bit_depth_conversion(self, src_fmt: str, to_fmt: str) -> bool:
        """Check if bit depth conversion is needed and compute shift value."""
        if to_fmt not in ("BGR888", "RGB888"):
            return False
        
        # Determine source bit depth from format string
        bit_depth = 16
        for cand in (16, 14, 12, 10):
            if str(cand) in src_fmt:
                bit_depth = cand
                break
        
        self._bit_depth_shift = max(0, bit_depth - 8)
        return self._bit_depth_shift > 0

    # ------------------------------------------------------------------
    # Optional logging helpers
    # ------------------------------------------------------------------

    def log_stream_options(self) -> None:
        """Log stream capabilities/options (on demand).

        - Output is sent to the log queue passed at construction time.
        - Only logs options relevant to the currently configured stream.
        """
        if  self._test_pattern:
            self._log(logging.INFO, "PiCam2:Test pattern mode active.")
            return

        if not self.cam_open or self.picam2 is None:
            self._log(logging.WARNING, "PiCam2:Camera not open; cannot list stream options")
            return

        if (self._stream_name or "").lower() == "main":
            if (self._output_width > 0) and (self._output_height > 0):
                main_size = (int(self._output_width), int(self._output_height))
            else:
                main_size = (int(self._capture_width), int(self._capture_height))

            supported = self.get_supported_main_color_formats()
            self._log(
                logging.INFO,
                f"PiCam2:Main Stream mode {main_size[0]}x{main_size[1]} format={self._format}. Supported main formats: {', '.join(supported)}",
            )
            self._log(
                logging.INFO,
                "PiCam2:Main Stream can scale to arbitrary resolutions; non-native aspect ratios may crop. For raw modes list, run examples/list_Picamera2Properties.py.",
            )

            try:
                main_opts = self.get_supported_main_options()
                if main_opts:
                    self._log(logging.INFO, "PiCam2:Suggested Main Stream options (camera_res/output_res, max_fps, full_fov):")
                    for opt in main_opts:
                        cr = opt.get("camera_res", (0, 0))
                        mr = opt.get("output_res", cr)
                        fmt = opt.get("main_format", "BGR888")
                        fps = float(opt.get("max_fps", 0.0) or 0.0)
                        full = bool(opt.get("full_fov", False))
                        self._log(
                            logging.INFO,
                            f"PiCam2:  {cr[0]}x{cr[1]} -> {mr[0]}x{mr[1]} fmt={fmt} max_fps~{fps:.1f} full_fov={full}",
                        )
            except Exception:
                pass
            return

        # RAW stream option dump
        if (self._stream_name or "").lower() == "raw":
            raw_size = (int(self._capture_width), int(self._capture_height))

            # Emit a note about potential FOV cropping in raw windowed modes
            try:
                props = getattr(self.picam2, "camera_properties", {})
                aw, ah = None, None
                if isinstance(props, dict):
                    pas = props.get("PixelArraySize")
                    if isinstance(pas, (list, tuple)) and len(pas) == 2:
                        aw, ah = int(pas[0]), int(pas[1])
                if aw and ah:
                    if raw_size[0] < aw and raw_size[1] < ah:
                        self._log(
                            logging.INFO,
                            f"PiCam2:RAW mode {raw_size[0]}x{raw_size[1]} is a sensor window (cropped FOV vs {aw}x{ah}).",
                        )
            except Exception:
                pass

            try:
                modes = self.get_supported_raw_options()
                if modes:
                    self._log(
                        logging.INFO,
                        f"PiCam2:Raw Stream {raw_size[0]}x{raw_size[1]} format={self._format}. Available RAW sensor modes (fmt@WxH~fps):",
                    )
                    for m in modes:
                        fmt = m.get("format")
                        size = m.get("size", (0, 0))
                        fps = float(m.get("fps", 0.0) or 0.0)
                        self._log(logging.INFO, f"PiCam2:  {fmt}@{size[0]}x{size[1]}~{fps:.1f}")
                    self._log(
                        logging.INFO,
                        "PiCam2:Raw Stream resolutions/formats must match sensor modes. See examples/list_Picamera2Properties.py for details.",
                    )
            except Exception:
                pass
            return

        self._log(logging.INFO, f"PiCam2:Unknown stream '{self._stream_name}'; no options to log")

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    # Logging helpers
    # ---------------

    def _log(self, level: int, msg: str) -> None:
        """Log a message to the log queue if available, with fallback if full."""
        q = self.log
        if q is None:
            return
        try:
            if not q.full():
                q.put_nowait((int(level), str(msg)))
        except Exception:
            # Fallback: print to stderr if log queue is full or unavailable
            import sys
            print(f"[LOG-{level}] {msg}", file=sys.stderr)

    def log_camera_config_and_controls(self) -> None:
        """Log current camera configuration and key timing/AE controls."""
        self._log(logging.INFO, "PiCam2:=== camera configuration ===")
        try:
            self._log(
                logging.INFO,
                "PiCam2:Requested mode={} camera_res={} output_res={} format={} fps={} stream_policy={} low_latency={} flip={}".format(
                    self._mode,
                    self._camera_res,
                    self._output_res,
                    self._requested_format or self._main_format,
                    self._framerate,
                    self._stream_policy,
                    self._low_latency,
                    self._flip_method,
                ),
            )
        except Exception:
            pass
        try:
            self._log(
                logging.INFO,
                "PiCam2:Requested controls exposure={} autoexposure={} aemeteringmode={} autowb={} awbmode={}".format(
                    self._exposure,
                    self._autoexposure,
                    self._configs.get("aemeteringmode", None),
                    self._autowb,
                    self._configs.get("awbmode", None),
                ),
            )
        except Exception:
            pass

        # Camera configuration / properties (best-effort)
        try:
            if self.picam2 is not None:
                cfg_fn = getattr(self.picam2, "camera_configuration", None)
                if callable(cfg_fn):
                    cfg_now = cfg_fn()
                    self._log(logging.INFO, f"PiCam2:camera_configuration={cfg_now}")
                    try:
                        cfg_controls = cfg_now.get("controls") if isinstance(cfg_now, dict) else None
                        if cfg_controls:
                            self._log(logging.INFO, f"PiCam2:configured controls={cfg_controls}")
                    except Exception:
                        pass
        except Exception as exc:
            self._log(logging.INFO, f"PiCam2:camera_configuration unavailable ({exc})")

        try:
            if self._last_set_controls:
                self._log(logging.INFO, f"PiCam2:last set_controls={self._last_set_controls}")
        except Exception:
            pass

        # Best-effort readback of current controls (if supported by Picamera2 build)
        try:
            if self.picam2 is not None:
                get_controls = getattr(self.picam2, "get_controls", None)
                cam_ctrls = getattr(self.picam2, "camera_controls", None)
                if callable(get_controls):
                    if isinstance(cam_ctrls, dict) and cam_ctrls:
                        names = list(cam_ctrls.keys())
                    else:
                        names = [
                            "AeEnable", 
                            "AeMeteringMode", 
                            "AwbEnable", 
                            "AwbMode", 
                            "ExposureTime", 
                            "AnalogueGain"
                            "AeEnable", #
                            "AeExposureMode", #
                            "AeFlickerPeriod",
                            "AeMeteringMode", #
                            "AfMode", #
                            "AnalogueGain", #
                            "AwbEnable", #
                            "AwbMode", #
                            "Brightness", #
                            "ColourGains", #
                            "ColourTemperature", #
                            "Contrast", #
                            "ExposureTime", #
                            "FrameDuration", #
                            "FrameDurationLimits", #
                            "LensPosition", #
                            "NoiseReductionMode", #
                            "Saturation", #
                            "ScalerCrop", #
                            "Sharpness", #
                        ]

                    try:
                        current = get_controls(names)
                    except Exception:
                        current = get_controls()
                    if isinstance(current, dict) and current:
                        self._log(logging.INFO, f"PiCam2:readback controls={current}")
                else:
                    self._log(logging.INFO, "PiCam2:readback controls unavailable (get_controls missing)")
                try:
                    if isinstance(cam_ctrls, dict) and cam_ctrls:
                        keys = sorted([str(k) for k in cam_ctrls.keys()])
                        self._log(logging.INFO, f"PiCam2:available controls ({len(keys)}): {keys}")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.picam2 is not None:
                props = getattr(self.picam2, "camera_properties", None)
                if props is not None:
                    self._log(logging.INFO, f"PiCam2:camera_properties={props}")
        except Exception:
            pass

        # Timing-related metadata
        try:
            md = {}
            if self.picam2 is not None:
                with self.cam_lock:
                    md = self._capture_metadata()
            if isinstance(md, dict) and md:
                fd = md.get("FrameDuration")
                fdl = md.get("FrameDurationLimits")
                sc = md.get("ScalerCrop")
                ae = md.get("AeEnable", None)
                exp = md.get("ExposureTime", None)
                awb_en = md.get("AwbEnable", None)
                awb_mode = md.get("AwbMode", None)
                ae_meter = md.get("AeMeteringMode", None)
                gain = md.get("AnalogueGain", None)
                self._log(
                    logging.INFO,
                    "PiCam2:metadata FrameDuration={} FrameDurationLimits={} ScalerCrop={} AeEnable={} ExposureTime={} AwbEnable={} AwbMode={} AeMeteringMode={} AnalogueGain={}".format(
                        fd,
                        fdl,
                        sc,
                        ae,
                        exp,
                        awb_en,
                        awb_mode,
                        ae_meter,
                        gain,
                    ),
                )
        except Exception as exc:
            self._log(logging.INFO, f"PiCam2:capture_metadata unavailable ({exc})")

    # Control and Format helpers
    # --------------------------

    def _get_control(self, name: str, force_refresh: bool = False):
        """Get a single control/metadata value by name. If force_refresh is True, always update metadata."""
        if force_refresh or self._metadata is None or name not in self._metadata:
            if self.picam2 is None:
                return None
            with self.cam_lock:
                self._metadata = self._capture_metadata()
        return (self._metadata or {}).get(name)

    def _set_controls(self, controls: dict) -> bool:
        """Set multiple controls on the camera."""
        if self.picam2 is None:
            return False
        if not controls:
            return True
        try:
            with self.cam_lock:
                self.picam2.set_controls(controls)
            try:
                self._last_set_controls = dict(controls)
            except Exception:
                self._last_set_controls = None
            return True
        except Exception as exc:
            self._log(logging.WARNING, f"PiCam2:Failed to set controls {controls}: {exc}")
            return False

    def _parse_aemeteringmode(self, value) -> int:
        """Parse AE metering mode from string or int to Picamera2 AeMeteringMode enum."""
        if value is None or value == -1:
            return 0
        if isinstance(value, str):
            key = value.strip().lower()
            if key in self._AE_METERING_MAP:
                return int(self._AE_METERING_MAP[key])
        try:
            return int(value)
        except Exception:
            return 0

    def _parse_awbmode(self, value) -> int:
        """Parse AWB mode from string or int to Picamera2 AwbMode enum."""
        if value is None or value == -1:
            return 0
        if isinstance(value, str):
            key = value.strip().lower()
            if key in self._AWB_MODE_MAP:
                return int(self._AWB_MODE_MAP[key])
        try:
            return int(value)
        except Exception:
            return 0

    def _map_fourcc_format(self, fourcc: str) -> str | None:
        """Map legacy FOURCC / format strings to Picamera2 format strings."""
        if not fourcc:
            return None
        if self._is_raw_format(fourcc):
            return fourcc
        fourcc_u = self._format_upper(fourcc)

        table = {
            "BGR3": "BGR888",
            "RGB3": "RGB888",
            "MJPG": "MJPEG",
            "YUY2": "YUYV",
            "YUYV": "YUYV",
            "YUV420": "YUV420",
            "YU12": "YUV420",
            "I420": "YUV420",
        }

        # Pass-through for already-valid Picamera2 main formats.
        passthrough = set(self._supported_main_color_formats())
        return table.get(fourcc_u) or (fourcc_u if fourcc_u in passthrough else None)

    @staticmethod
    def _format_upper(fmt) -> str:
        """Best-effort format string normalizer."""
        if fmt is None:
            return ""
        if isinstance(fmt, str):
            return fmt.upper()
        if isinstance(fmt, (bytes, bytearray)):
            try:
                return bytes(fmt).decode("utf-8", errors="ignore").upper()
            except Exception:
                return ""

        name = getattr(fmt, "name", None)
        if isinstance(name, str):
            return name.upper()

        # Fallback: string representation of libcamera/Picamera2 objects.
        try:
            return str(fmt).upper()
        except Exception:
            return ""

    @staticmethod
    def _bayer_prefix(fmt_str: str) -> str:
        """Return the Bayer prefix (SRGGB/SBGGR/SGBRG/SGRBG) or '' if not Bayer."""
        if not fmt_str:
            return ""
        for p in ("SRGGB", "SBGGR", "SGBRG", "SGRBG"):
            if fmt_str.startswith(p):
                return p
        return ""

    def _is_raw_format(self, fmt) -> bool:
        """Check if the format string represents a raw Bayer format."""
        fmt_str = self._format_upper(fmt)
        return bool(self._bayer_prefix(fmt_str))

    def _list_sensor_modes(self) -> list[dict]:
        """List all sensor modes from Picamera2, normalized.

        Returns dicts with keys:
        format, size=(w,h), fps, area, bit_depth, crop_limits, crop_area
        """
        modes: list[dict] = []

        if self.picam2 is None:
            return modes

        try:
            sensor_modes = getattr(self.picam2, "sensor_modes", None)
            if not isinstance(sensor_modes, list):
                return modes

            for m in sensor_modes:
                try:
                    fmt = m.get("format")
                    size_val = m.get("size")
                    if not fmt or not isinstance(size_val, (list, tuple)) or len(size_val) != 2:
                        continue

                    w, h = int(size_val[0]), int(size_val[1])

                    fps = m.get("fps", m.get("max_fps", 0))
                    try:
                        fps = float(fps) if fps is not None else 0.0
                    except Exception:
                        fps = 0.0

                    bit_depth = m.get("bit_depth", None)
                    try:
                        bit_depth = int(bit_depth) if bit_depth is not None else None
                    except Exception:
                        bit_depth = None

                    crop_limits = m.get("crop_limits", None)
                    crop_area = None
                    if isinstance(crop_limits, (list, tuple)) and len(crop_limits) == 4:
                        # crop_limits is (x, y, w, h)
                        try:
                            crop_area = int(crop_limits[2]) * int(crop_limits[3])
                        except Exception:
                            crop_area = None

                    modes.append(
                        {
                            "format": str(fmt).upper(),
                            "size": (w, h),
                            "fps": fps,
                            "area": w * h,
                            "bit_depth": bit_depth,
                            "crop_limits": crop_limits,
                            "crop_area": crop_area,
                        }
                    )
                except Exception:
                    continue
        except Exception:
            pass

        return modes
        
    def _list_raw_sensor_modes(self) -> list[dict]:
        """List supported raw sensor modes in a normalized dict format.

        This is the shared source for:
        - get_supported_raw_options() (public)
        - _select_raw_sensor_mode() (selection)
        - _validate_raw_selection() (via get_supported_raw_options)
        """
        raw_modes: list[dict] = []
        for m in self._list_sensor_modes():
            try:
                fmt = m.get("format")
                size = m.get("size")
                if not fmt or not size or not self._is_raw_format(fmt):
                    continue
                fmt_str = str(fmt).upper()
                w, h = int(size[0]), int(size[1])
                fps = float(m.get("fps", 0.0) or 0.0)
                area = int(m.get("area", w * h))
                raw_modes.append({"format": fmt_str, "size": (w, h), "fps": fps, "area": area})
            except Exception:
                continue
        return raw_modes

    def _select_sensor_mode(self, modes: list[dict], desired_size: tuple[int, int] | None = None):
        """Select the best sensor mode from a provided list.

        Uses self._stream_policy to guide selection:
        - maximize_fps_no_crop / maximize_fps_with_crop: prioritize highest fps, then FOV (crop_area), then output area
        - maximize_fov: prioritize largest FOV (crop_area), then fps, then output area

        Shared by raw/main selection; callers may filter `modes` beforehand.
        """
        if not modes:
            return None

        def mode_area(m: dict) -> int:
            try:
                a = m.get("area")
                if a is not None:
                    return int(a)
                w, h = m.get("size", (0, 0))
                return int(w) * int(h)
            except Exception:
                return 0

        def fov_area(m: dict) -> int:
            """Prefer crop_area (true FOV proxy), fall back to output size area."""
            ca = m.get("crop_area")
            if ca is not None:
                try:
                    return int(ca)
                except Exception:
                    pass
            return mode_area(m)

        def mode_fps(m: dict) -> float:
            try:
                v = m.get("fps", 0.0)
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        desired_w, desired_h = (None, None)
        if desired_size and len(desired_size) == 2:
            try:
                desired_w, desired_h = int(desired_size[0]), int(desired_size[1])
            except Exception:
                desired_w, desired_h = (None, None)

        # --- Policy scorers ---
        def score_maximize_fps(m: dict) -> tuple:
            # 1) highest fps
            # 2) larger FOV (crop_area) as tie-break
            # 3) larger sensor output size as further tie-break
            return (-mode_fps(m), -fov_area(m), -mode_area(m))

        def score_maximize_fov(m: dict) -> tuple:
            # 1) largest FOV
            # 2) highest fps
            # 3) larger sensor output size
            return (-fov_area(m), -mode_fps(m), -mode_area(m))

        # Sort by the chosen policy
        if self._stream_policy == "maximize_fov":
            modes.sort(key=score_maximize_fov)
        else:
            # default behavior == maximize_fps policies
            modes.sort(key=score_maximize_fps)

        # Optional: desired_size proximity
        #
        # IMPORTANT: For maximize_fps policies, we generally do NOT want desired_size proximity
        # to override sensor-mode choice (it can pick a slower mode).
        # So we only apply it for non-maximize_fps policies.
        if desired_w and desired_h and self._stream_policy not in ("maximize_fps_no_crop", "maximize_fps_with_crop"):
            desired_area = desired_w * desired_h

            def add_area_penalty(m: dict) -> int:
                return abs(mode_area(m) - desired_area)

            # Stable sort keeps previous ordering as a tie-breaker.
            modes.sort(key=add_area_penalty)

        return modes[0]

    def _select_raw_sensor_mode(self, desired_size: tuple[int, int] | None = None):
        """Select the best raw sensor mode based on policy and desired size."""
        return self._select_sensor_mode(self._list_raw_sensor_modes(), desired_size=desired_size)

    def _select_main_sensor_mode(self, desired_size: tuple[int, int] | None = None):
        """Select the best main sensor mode based on policy and desired size."""
        return self._select_sensor_mode(self._list_sensor_modes(), desired_size=desired_size)

    def _validate_raw_selection(self, desired_fmt: str, desired_size: tuple[int, int]):
        """Validate and adjust desired raw format and size based on supported modes."""
        modes = self.get_supported_raw_options()
        if not modes:
            self._log(logging.INFO, "PiCam2:Raw modes unavailable; run examples/list_Picamera2Properties.py to list sensor modes.")
            return (desired_fmt, desired_size)

        desired_fmt = self._format_upper(desired_fmt)
        desired_prefix = self._bayer_prefix(desired_fmt)

        exact_sizes = []
        for m in modes:
            try:
                mf = m.get("format") or ""
                if mf == desired_fmt:
                    exact_sizes.append(m["size"])
            except Exception:
                continue
        if exact_sizes:
            if tuple(desired_size) in exact_sizes:
                return (desired_fmt, tuple(desired_size))
            sel_size = exact_sizes[0]
            self._log(logging.INFO, f"PiCam2:Requested raw size {desired_size} not in {desired_fmt}; using {sel_size}.")
            return (desired_fmt, sel_size)

        if desired_prefix:
            prefix_matches = []
            for m in modes:
                try:
                    mf = m.get("format") or ""
                    if self._bayer_prefix(self._format_upper(mf)) == desired_prefix:
                        prefix_matches.append(m)
                except Exception:
                    continue
            if prefix_matches:
                sizes_for_prefix = [m["size"] for m in prefix_matches]
                if tuple(desired_size) in sizes_for_prefix:
                    return (prefix_matches[sizes_for_prefix.index(tuple(desired_size))]["format"], tuple(desired_size))
                sel = prefix_matches[0]
                sel_fmt, sel_size = sel["format"], sel["size"]
                self._log(
                    logging.INFO,
                    f"PiCam2:Requested raw format {desired_fmt} not available; using {sel_fmt}@{sel_size[0]}x{sel_size[1]} (matched Bayer pattern {desired_prefix}).",
                )
                return (sel_fmt, sel_size)

        sel_fmt = modes[0]["format"]
        sel_size = modes[0]["size"]
        self._log(logging.INFO, f"PiCam2:Requested raw format {desired_fmt} not available; using {sel_fmt}@{sel_size[0]}x{sel_size[1]}")
        return (sel_fmt, sel_size)


    def _flip_to_transform(self, flip: int, Transform):
        """Convert flip_method integer to libcamera Transform object."""
        try:
            t = Transform()
        except Exception:
            return None
        if flip == 0:
            return t
        if flip == 1:  # ccw 90
            t = Transform(hflip=0, vflip=0, rotation=270)
        elif flip == 2:  # 180
            t = Transform(hflip=1, vflip=1)
        elif flip == 3:  # cw 90
            t = Transform(hflip=0, vflip=0, rotation=90)
        elif flip == 4:  # horizontal
            t = Transform(hflip=1, vflip=0)
        elif flip == 6:  # vertical
            t = Transform(hflip=0, vflip=1)
        elif flip == 5:  # upright diagonal
            t = Transform(hflip=1, vflip=0, rotation=270)
        elif flip == 7:  # upper-left diagonal
            t = Transform(hflip=1, vflip=0, rotation=90)
        return t

    # Camera frame buffer
    # -------------------
    @property
    def buffer(self):
        return self._buffer

    # Camera size/resolution
    # ----------------------

    @property
    def width(self):
        return int(self._capture_width)

    @width.setter
    def width(self, value):
        if value is None or value == -1:
            return
        self.size = (int(value), int(self._capture_height))

    @property
    def height(self):
        return int(self._capture_height)

    @height.setter
    def height(self, value):
        if value is None or value == -1:
            return
        self.size = (int(self._capture_width), int(value))

    @property
    def resolution(self):
        return (int(self._capture_width), int(self._capture_height))

    @resolution.setter
    def resolution(self, value):
        if value is None or value == -1:
            return
        if isinstance(value, (tuple, list)):
            if len(value) < 2:
                self.size = (int(value[0]), int(value[0]))
            else:
                self.size = (int(value[0]), int(value[1]))
        else:
            self.size = (int(value), int(value))

    @property
    def size(self):
        return (self._capture_width, self._capture_height)

    @size.setter
    def size(self, value):
        """Set camera size (width, height). If camera is open and not capturing, will close and reopen. Logs this action."""
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
            self._log(20, f"Camera open and not capturing: reopening camera to apply new size {width}x{height}.")
            self.close_cam()
            self.open_cam()

    # Camera flip/rotation
    # -------------------

    @property
    def flip(self) -> int:
        """Flip/rotation enum (0..7), same convention as cv2Capture."""
        try:
            return int(self._flip_method)
        except Exception:
            return 0

    @flip.setter
    def flip(self, value) -> None:
        """Set camera flip/rotation. If camera is open and not capturing, will close and reopen. Logs this action."""
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

        if self.cam_open:
            self._log(20, f"Camera open and not capturing: reopening camera to apply new flip {f}.")
            self.close_cam()
            self.open_cam()

    # Camera color format and conversion
    # ---------------------------------

    @property
    def convert_format(self) -> str:
        """Target format for automatic conversion (default: BGR888)."""
        return self._convert_format

    @convert_format.setter
    def convert_format(self, value: str):
        """Set target conversion format and update conversion settings."""
        if value is None:
            return
        try:
            fmt = str(value).upper()
        except Exception:
            return
        
        if fmt != self._convert_format:
            self._convert_format = fmt
            self._update_conversion_settings()

    # Camera Controls
    # ---------------

    @property
    def metadata(self):
        md = getattr(self, "_metadata", None)
        if md is not None:
            return md
        if self.picam2 is None:
            return {}
        with self.cam_lock:
            self._metadata = self._capture_metadata()
        return self._metadata or {}

    def _capture_metadata(self) -> dict:
        """Obtain metadata from Picamera2.

        Caller must hold self.cam_lock.
        """
        if self.picam2 is None:
            return {}
        try:
            capture_metadata = getattr(self.picam2, "capture_metadata", None)
            if callable(capture_metadata):
                metadata = capture_metadata()
                return dict(metadata) if metadata is not None else {}
        except Exception:
            return {}
        return {}


    @property
    def exposure(self):
        return self._get_control("ExposureTime")

    @exposure.setter
    def exposure(self, value):
        if value is None or value == -1:
            return
        self._exposure = float(value)
        self._set_controls({"AeEnable": False, "ExposureTime": int(value)})

    @property
    def gain(self):
        return self._get_control("AnalogueGain")

    @gain.setter
    def gain(self, value):
        if value is None:
            return
        self._set_controls({"AnalogueGain": float(value)})

    @property
    def autoexposure(self):
        ae = self._get_control("AeEnable")
        if ae is None:
            return -1
        return 1 if bool(ae) else 0

    @autoexposure.setter
    def autoexposure(self, value):
        if value is None or value == -1:
            return
        self._autoexposure = 1 if bool(value) else 0
        self._set_controls({"AeEnable": bool(value)})

    @property
    def aemeteringmode(self):
        val = self._get_control("AeMeteringMode")
        if val is None:
            return 0
        return int(val)

    @aemeteringmode.setter
    def aemeteringmode(self, value):
        meter_val = self._parse_aemeteringmode(value)
        self._set_controls({"AeMeteringMode": meter_val})

    @property
    def aeexposuremode(self):
        return self._get_control("AeExposureMode")

    @aeexposuremode.setter
    def aeexposuremode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AeExposureMode": int(value)})

    @property
    def autowb(self):
        awb = self._get_control("AwbEnable")
        if awb is None:
            return -1
        return 1 if bool(awb) else 0

    @autowb.setter
    def autowb(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AwbEnable": bool(value)})

    @property
    def awbmode(self):
        val = self._get_control("AwbMode")
        if val is None:
            return 0
        return int(val)

    @awbmode.setter
    def awbmode(self, value):
        awb_val = self._parse_awbmode(value)
        self._set_controls({"AwbMode": awb_val})

    @property
    def wbtemperature(self):
        return self._get_control("ColourTemperature")

    @wbtemperature.setter
    def wbtemperature(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"ColourTemperature": int(value)})

    @property
    def colourgains(self):
        return self._get_control("ColourGains")

    @colourgains.setter
    def colourgains(self, value):
        if value is None:
            return
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("ColourGains must be a (red_gain, blue_gain) tuple")
        self._set_controls({"ColourGains": (float(value[0]), float(value[1]))})

    @property
    def afmode(self):
        return self._get_control("AfMode")

    @afmode.setter
    def afmode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AfMode": int(value)})

    @property
    def lensposition(self):
        return self._get_control("LensPosition")

    @lensposition.setter
    def lensposition(self, value):
        if value is None:
            return
        self._set_controls({"LensPosition": float(value)})

    @property
    def brightness(self):
        return self._get_control("Brightness")

    @brightness.setter
    def brightness(self, value):
        if value is None:
            return
        self._set_controls({"Brightness": float(value)})

    @property
    def contrast(self):
        return self._get_control("Contrast")

    @contrast.setter
    def contrast(self, value):
        if value is None:
            return
        self._set_controls({"Contrast": float(value)})

    @property
    def saturation(self):
        return self._get_control("Saturation")

    @saturation.setter
    def saturation(self, value):
        if value is None:
            return
        self._set_controls({"Saturation": float(value)})

    @property
    def sharpness(self):
        return self._get_control("Sharpness")

    @sharpness.setter
    def sharpness(self, value):
        if value is None:
            return
        self._set_controls({"Sharpness": float(value)})

    @property
    def noisereductionmode(self):
        return self._get_control("NoiseReductionMode")

    @noisereductionmode.setter
    def noisereductionmode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"NoiseReductionMode": int(value)})

    @property
    def fps(self):
        # Prefer deriving from FrameDuration (us) when available.
        try:
            fd = self._get_control("FrameDuration")
            if fd is not None:
                fd_us = float(fd)
                if fd_us > 0:
                    return 1_000_000.0 / fd_us
        except Exception:
            pass

        # Next, try FrameDurationLimits (min_us, max_us)
        try:
            fdl = self._get_control("FrameDurationLimits")
            if isinstance(fdl, (list, tuple)) and len(fdl) == 2:
                fd_us = float(fdl[0])
                if fd_us > 0:
                    return 1_000_000.0 / fd_us
        except Exception:
            pass

        # Fallback: some stacks expose FrameRate
        fps = self._get_control("FrameRate")
        if fps is None:
            return float(self._framerate)
        try:
            return float(fps)
        except Exception:
            return float(self._framerate)

    @fps.setter
    def fps(self, value):
        if value is None or value == -1:
            return
        try:
            fr = float(value)
        except Exception:
            return
        self._framerate = fr
        if fr <= 0:
            return
        # Prefer FrameDurationLimits for consistency with direct script.
        try:
            frame_us = int(round(1_000_000.0 / fr))
            if frame_us > 0:
                self._set_controls({"FrameDurationLimits": (frame_us, frame_us)})
                return
        except Exception:
            pass
        # Best-effort fallback.
        self._set_controls({"FrameRate": float(fr)})

__all__ = ["PiCamera2Core"]
