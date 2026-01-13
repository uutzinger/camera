###############################################################################
# Raspberry Pi CSI camera core (Picamera2 / libcamera)
#
# Shared core used by both non-Qt threaded capture and Qt wrappers.
#
# Urs Utzinger
# GPT-5.2
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: PiCamera2Core
#
# This is a non-threaded, non-Qt core wrapper intended to be composed by:
# - the threaded capture wrapper `piCamera2Capture`
# - the Qt wrapper `piCamera2CaptureQt`
#
# Public methods:
# - open_cam()
# - close_cam()
# - capture_array() -> tuple[np.ndarray | None, float | None]
# - convert(frame, to: str = 'BGR888') -> np.ndarray | None
# - postprocess(frame) -> np.ndarray | None
# - get_metadata() -> dict
# - get_control(name: str) -> Any
# - set_controls(controls: dict) -> bool
# - get_supported_main_color_formats() -> list[str]
# - get_supported_raw_color_formats() -> list[str]
# - get_supported_raw_options() -> list[dict]
# - get_supported_main_options() -> list[dict]
# - log_stream_options() -> None
#
# Convenience properties:
# - cam_open: bool (state flag)
# - capturing: bool
# - metadata: dict
# - width / height / resolution / size 
#       (note: changing size/flip while capturing is discouraged;
#       wrappers should ensure capture is paused first)
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
#                                  in raw mode, this may trigger CPU resize
# - fps: float|int                 Requested framerate
# - displayfps: float|int          Qt wrapper legacy/ignored (emission is unthrottled)
# - low_latency: bool              Hint to reduce buffering (core may request
#                                  fewer libcamera buffers)
#
# Stream selection:
# - mode: str                      'main' (processed) or 'raw' (sensor Bayer)
# - stream_policy: str             'default'|'maximize_fov'|'maximize_fps'
#
# Formats:
# - format: str                    Legacy/combined request (e.g. 'BGR3', 'YUY2', 'SRGGB8')
# - fourcc: str                    Legacy FOURCC alternative
# - main_format: str               Explicit main stream format override
# - raw_format: str                Explicit raw Bayer format override
# - raw_res: tuple[int,int]        Requested raw sensor window size (w,h)
#
# Controls (applied at open, and/or via properties/set_controls):
# - exposure: int|float            Manual exposure in microseconds; <=0 leaves AE enabled
# - autoexposure: int              -1 leave unchanged, 0 AE off, 1 AE on
# - aemeteringmode: int|str        0/1/2 or 'center'|'spot'|'matrix'
# - autowb: int                    -1 leave unchanged, 0 AWB off, 1 AWB on
# - awbmode: int|str               0..5 or 'auto'|'tungsten'|'fluorescent'|'indoor'|'daylight'|'cloudy'
#
# Low-level / advanced:
# - buffer_count: int|None         Requested libcamera buffer_count (if supported)
# - hw_transform: bool             Attempt libcamera hardware Transform for flip/rotation
# - flip: int                      0..7 (same enum as cv2Capture)
###############################################################################

from __future__ import annotations

from threading import Lock
from queue import Queue         # logging
import logging
import time

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None  # type: ignore
from typing import Optional, Tuple


class FrameBuffer:
    """
    Single-producer / single-consumer (SPSC) ring buffer for fixed-shape NumPy frames.

    Notes:
      - Uses power-of-two capacity for fast masking.
      - Stores frames in a preallocated ndarray: (capacity, *frame_shape).
      - Can operate with overwrite=False (default): push fails if full.
        If overwrite=True: oldest frames are dropped when full (ring overwrite).

    init parameters:
        - capacity: number of frames the ring can hold (rounded up to power of two)
        - frame_shape: shape of one frame, e.g. (H, W) or (H, W, C)
        - dtype: numpy dtype for storage
        - overwrite: if True, newest data overwrites oldest when full
    
    methods:
        - push(frame: np.ndarray, ts_ms: float) -> bool
        - pull(copy: bool = True) -> tuple[Optional[np.ndarray], Optional[float]]
        - pull_batch(max_items: int, copy: bool = True, out: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]
        - clear() -> None
        - avail() -> int
        - free() -> int
        - capacity() -> int
        - frame_shape() -> Tuple[int, ...]
        - dtype() -> np.dtype
    """

    __slots__ = (
        "_buf",
        "_buf_ts_ms",
        "_cap",
        "_mask",
        "_shape",
        "_dtype",
        "_head",
        "_tail",
        "_overwrite",
        "_scratch",
        "_scratch_ts",
    )

    def __init__(
        self,
        capacity: int,
        frame_shape: Tuple[int, ...],
        dtype: np.dtype | str = np.uint8,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        capacity: number of frames the ring can hold (rounded up to power of two)
        frame_shape: shape of one frame, e.g. (H, W) or (H, W, C)
        dtype: numpy dtype for storage
        overwrite: if True, newest data overwrites oldest when full
        aligned: if True, attempts to allocate aligned storage (usually unnecessary)
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not frame_shape:
            raise ValueError("frame_shape must be a non-empty tuple")

        cap = 1 << (int(capacity - 1).bit_length())  # next power of two
        self._cap = cap
        self._mask = cap - 1
        self._shape = tuple(int(x) for x in frame_shape)
        self._dtype = np.dtype(dtype)
        self._overwrite = bool(overwrite)

        # Preallocate buffer: (cap, *frame_shape)
        self._buf = np.empty((cap, *self._shape), dtype=self._dtype, order="C")
        self._buf_ts_ms = np.empty((cap,), dtype=np.float64, order="C")

        # Monotonic counters (unbounded ints). Only producer writes _head; only consumer writes _tail.
        self._head = 0
        self._tail = 0

        # Scratch buffer for pull_batch to return a contiguous block without realloc each time (optional use)
        self._scratch: Optional[np.ndarray] = None
        self._scratch_ts: Optional[np.ndarray] = None

    # ----------------------------
    # Introspection / sizing
    # ----------------------------
    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def avail(self) -> int:
        """How many frames are currently available to read."""
        head = self._head
        tail = self._tail
        if self._overwrite:
            min_tail = head - self._cap
            if tail < min_tail:
                tail = min_tail
        n = head - tail
        if n < 0:
            return 0
        if n > self._cap:
            return self._cap
        return n

    def free(self) -> int:
        """How many free slots remain (0..capacity)."""
        free = self._cap - self.avail()
        return 0 if free < 0 else free

    # ----------------------------
    # Producer side
    # ----------------------------
    def push(self, frame: np.ndarray, ts_ms: float | int | np.floating) -> bool:
        """
        Push one frame.
        Returns True if stored, False if dropped (when full and overwrite=False).
        """
        # Fast locals
        head = self._head
        tail = self._tail
        cap = self._cap

        used = head - tail
        if used >= cap:
            if not self._overwrite:
                return False
            # Overwrite mode:
            # Do NOT advance tail here (producer must not write consumer-owned tail).
            # Instead, let the consumer detect overrun and skip forward.

        idx = head & self._mask
        slot = self._buf[idx]

        # Copy data into slot (avoids allocating new arrays).
        # np.copyto is usually fast and supports dtype conversion if needed.
        np.copyto(slot, frame, casting="unsafe")
        try:
            self._buf_ts_ms[idx] = float(ts_ms)
        except Exception:
            self._buf_ts_ms[idx] = 0.0

        # Publish: increment head ONLY after copy completes
        self._head = head + 1
        return True

    # ----------------------------
    # Consumer side
    # ----------------------------
    def pull(self, *, copy: bool = True) -> tuple[Optional[np.ndarray], Optional[float]]:
        """
        Pull one frame.
        If copy=True (default), returns a copy safe from being overwritten later.
        If copy=False, returns a view into the ring slot (FAST, but unsafe if producer overwrites).
        """
        tail = self._tail
        head = self._head

        # If producer has overrun the consumer (overwrite mode), skip to the
        # oldest still-valid element.
        if self._overwrite:
            min_tail = head - self._cap
            if tail < min_tail:
                tail = min_tail
                self._tail = tail

        if tail >= head:
            return None, None

        idx = tail & self._mask
        frame_view = self._buf[idx]
        ts = float(self._buf_ts_ms[idx])
        self._tail = tail + 1

        if copy:
            return frame_view.copy(), ts
        else:
            return frame_view, ts

    def pull_batch(
        self,
        max_items: int,
        *,
        copy: bool = True,
        out: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Pull up to max_items frames.
        Returns an ndarray of shape (n, *frame_shape). n may be 0.

        copy=True: returns a contiguous copy (safe).
        copy=False: returns a stacked copy anyway (because a true view would be non-contiguous when wrapped).
                   If you want zero-copy, pull(copy=False) in a loop and process immediately.

        out: optional preallocated output array with shape (max_items, *frame_shape) and matching dtype.
             If provided, returns out[:n].
        """
        if max_items <= 0:
            return (
                np.empty((0, *self._shape), dtype=self._dtype),
                np.empty((0,), dtype=np.float64),
            )

        tail = self._tail
        head = self._head

        # If producer has overrun the consumer (overwrite mode), skip to the
        # oldest still-valid element.
        if self._overwrite:
            min_tail = head - self._cap
            if tail < min_tail:
                tail = min_tail
                self._tail = tail
        available = head - tail
        if available <= 0:
            return (
                np.empty((0, *self._shape), dtype=self._dtype),
                np.empty((0,), dtype=np.float64),
            )

        n = available if available < max_items else max_items
        cap = self._cap
        mask = self._mask
        buf = self._buf

        if out is not None:
            if out.shape[:1] != (max_items,) or out.shape[1:] != self._shape:
                raise ValueError(f"out must have shape ({max_items}, {self._shape}), got {out.shape}")
            if out.dtype != self._dtype:
                raise ValueError(f"out dtype must be {self._dtype}, got {out.dtype}")
            dst = out
        else:
            # Reuse scratch if possible to avoid repeated allocations for same max_items
            if self._scratch is None or self._scratch.shape[0] < n or self._scratch.dtype != self._dtype:
                self._scratch = np.empty((max_items, *self._shape), dtype=self._dtype)
            dst = self._scratch

        if self._scratch_ts is None or self._scratch_ts.shape[0] < max_items:
            self._scratch_ts = np.empty((max_items,), dtype=np.float64)
        dst_ts = self._scratch_ts

        # Copy in at most two chunks (no wrap + wrap)
        start = tail & mask
        first = min(n, cap - start)
        second = n - first

        # Copy chunk(s)
        dst0 = dst[:first]
        np.copyto(dst0, buf[start:start + first])
        dst_ts[:first] = self._buf_ts_ms[start:start + first]

        if second:
            dst1 = dst[first:first + second]
            np.copyto(dst1, buf[0:second])
            dst_ts[first:first + second] = self._buf_ts_ms[0:second]

        # Advance tail after copies complete
        self._tail = tail + n

        # Return exact-size view
        batch = dst[:n]
        ts_batch = dst_ts[:n]
        return batch, ts_batch

    def clear(self) -> None:
        """Drop all pending frames."""
        self._tail = self._head


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

        # Used by wrappers to prevent risky live reconfiguration
        self._capturing = False

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

        except Exception:
            pass

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
        self._stream_policy = str(self._configs.get("stream_policy", "default")).lower()
        self._main_format = str(self._configs.get("main_format", self._requested_format))
        self._raw_format = str(self._configs.get("raw_format", self._requested_format))
        self._raw_res = self._configs.get("raw_res", (self._capture_width, self._capture_height))
        self._low_latency = bool(self._configs.get("low_latency", False))
        self._buffer_count = self._configs.get("buffer_count", None)
        self._hw_transform = bool(self._configs.get("hw_transform", 1))

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
        if self.picam2 is None:
            return None, None

        try:
            with self.cam_lock:
                img = self.picam2.capture_array(self._stream_name)
                self._metadata = self._capture_metadata()
            ts_ms = None
            try:
                md = self._metadata or {}
                sensor_ts = md.get("SensorTimestamp")
                if sensor_ts is not None:
                    # Manual: nanoseconds since boot -> milliseconds
                    ts_ms = float(sensor_ts) / 1_000_000.0
                else:
                    # Fallback for older/alternate pipelines
                    alt = md.get("Timestamp")
                    if alt is not None:
                        ts_ms = float(alt)
            except Exception:
                ts_ms = time.perf_counter() * 1000.0
            return img, ts_ms

        except Exception as exc:
            self._log(logging.WARNING, f"PiCam2:Capture failed: {exc}")
            return None, None

    def postprocess(self, frame):
        """Apply drop-alpha, optional CPU resize, and optional CPU flip."""
        if frame is None:
            return None
        img_proc = frame

        # If the configured format yields 4 channels, drop alpha.
        if self._needs_drop_alpha:
            try:
                if getattr(img_proc, "ndim", 0) == 3 and img_proc.shape[2] == 4:
                    img_proc = img_proc[:, :, :3]
            except Exception:
                pass

        # Resize only when software resize is required (raw streams with output_res)
        if self._needs_cpu_resize:
            try:
                img_proc = cv2.resize(img_proc, self._output_res)
            except Exception:
                pass

        # Apply flip/rotation if requested (same enum as cv2Capture)
        if self._needs_cpu_flip:
            try:
                if self._flip_method == 1:  # ccw 90
                    img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 2:  # rot 180
                    img_proc = cv2.rotate(img_proc, cv2.ROTATE_180)
                elif self._flip_method == 3:  # cw 90
                    img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_CLOCKWISE)
                elif self._flip_method == 4:  # horizontal (left-right)
                    img_proc = cv2.flip(img_proc, 1)
                elif self._flip_method == 5:  # upright diagonal. ccw & lr
                    img_proc = cv2.flip(cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                elif self._flip_method == 6:  # vertical (up-down)
                    img_proc = cv2.flip(img_proc, 0)
                elif self._flip_method == 7:  # upperleft diagonal
                    img_proc = cv2.transpose(img_proc)
            except Exception:
                pass

        return img_proc

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open_cam(self) -> bool:
        """Open and configure the Picamera2 camera.

        This method configures the requested stream (main/raw), applies
        transform where possible, starts the camera, and sets initial controls.
        Logging is sent to the optional queue passed as `log_queue`.
        """
        if Picamera2 is None:
            self._log(logging.CRITICAL, "PiCam2:picamera2 is not installed")
            self.cam_open = False
            return False

        try:
            self.picam2 = Picamera2(camera_num=self.camera_num)

            req = str(self._requested_format or self._fourcc or "").upper()

            if self._mode == "raw":
                self._stream_name = "raw"
                rf = self._raw_format if self._is_raw_format(self._raw_format) else (
                    req if self._is_raw_format(req) else "SRGGB8"
                )
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
            else:
                self._stream_name = "main"
                pfmt = self._main_format or req
                mapped = self._map_fourcc_format(pfmt) or "BGR888"
                if mapped not in self.get_supported_main_color_formats():
                    self._log(
                        logging.INFO,
                        f"PiCam2:Main format {mapped} not ideal; using BGR888. Supported main formats: {', '.join(self.get_supported_main_color_formats())}",
                    )
                    self._log(logging.INFO, "PiCam2:For raw formats and resolutions, run examples/list_Picamera2Properties.py")
                    mapped = "BGR888"
                picam_format = mapped

            self._set_color_format(picam_format)
            picam_format = self._format or picam_format

            if (self._output_width > 0) and (self._output_height > 0):
                main_size = (self._output_width, self._output_height)
            else:
                main_size = (self._capture_width, self._capture_height)

            if self._stream_name != "raw":
                try:
                    raw_w, raw_h = int(self._raw_res[0]), int(self._raw_res[1])
                    raw_size = (raw_w, raw_h)
                except Exception:
                    raw_size = (self._capture_width, self._capture_height)

            transform = None
            if self._hw_transform and (self._flip_method != 0):
                try:
                    from libcamera import Transform  # type: ignore

                    transform = self._flip_to_transform(self._flip_method, Transform)
                except Exception:
                    transform = None

            transform_applied = transform is not None and self._flip_method != 0
            self._needs_cpu_flip = (self._flip_method != 0) and (not transform_applied) and (not self._is_yuv420)

            config = None
            try:
                if self._stream_name == "raw":
                    picam_kwargs = dict(raw={"size": raw_size, "format": picam_format}, controls={"FrameRate": self._framerate})
                else:
                    picam_kwargs = dict(main={"size": main_size, "format": picam_format}, controls={"FrameRate": self._framerate})

                buffer_count = self._buffer_count
                if buffer_count is None and self._low_latency:
                    buffer_count = 3
                if buffer_count is not None:
                    try:
                        picam_kwargs["buffer_count"] = int(buffer_count)
                    except Exception:
                        pass
                if transform is not None:
                    picam_kwargs["transform"] = transform

                config = self.picam2.create_video_configuration(**picam_kwargs)
            except Exception:
                if self._stream_name == "raw":
                    for fmt in ("SRGGB8", "SRGGB10_CSI2P"):
                        try:
                            picam_kwargs = dict(raw={"size": raw_size, "format": fmt}, controls={"FrameRate": self._framerate})
                            if transform is not None:
                                picam_kwargs["transform"] = transform
                            config = self.picam2.create_video_configuration(**picam_kwargs)
                            self._set_color_format(fmt)
                            break
                        except Exception:
                            continue
                if config is None:
                    for fmt in ("BGR888", "RGB888"):
                        try:
                            picam_kwargs = dict(main={"size": main_size, "format": fmt}, controls={"FrameRate": self._framerate})
                            if transform is not None:
                                picam_kwargs["transform"] = transform
                            config = self.picam2.create_video_configuration(**picam_kwargs)
                            self._set_color_format(fmt)
                            self._stream_name = "main"
                            break
                        except Exception:
                            continue

            self.picam2.configure(config)

            self._needs_cpu_resize = (
                (self._stream_name == "raw")
                and (self._output_width > 0)
                and (self._output_height > 0)
                and (not self._is_yuv420)
            )

            controls = {}

            if self._stream_name == "main":
                try:
                    props = getattr(self.picam2, "camera_properties", {})
                    crop_rect = None
                    paa = None
                    if isinstance(props, dict):
                        paa = props.get("PixelArrayActiveAreas") or props.get("ActiveArea")
                    if paa:
                        rect = None
                        if isinstance(paa, (list, tuple)):
                            rect = paa[0] if len(paa) > 0 else None
                        else:
                            rect = paa
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

            exposure = self._exposure
            autoexp = self._autoexposure
            autowb = self._autowb

            manual_requested = exposure is not None and exposure > 0
            if manual_requested:
                controls["AeEnable"] = False
                controls["ExposureTime"] = int(exposure)
            else:
                if autoexp is None or autoexp == -1:
                    pass
                elif autoexp > 0:
                    controls["AeEnable"] = True
                else:
                    controls["AeEnable"] = False

            try:
                cfg_meter = self._configs.get("aemeteringmode", 0)
                meter_val = self._parse_aemeteringmode(cfg_meter)
                controls.setdefault("AeMeteringMode", meter_val)
            except Exception:
                pass

            if autowb is None or autowb == -1:
                pass
            else:
                controls["AwbEnable"] = bool(autowb)

            try:
                cfg_awbmode = self._configs.get("awbmode", 0)
                awbmode_val = self._parse_awbmode(cfg_awbmode)
                controls.setdefault("AwbMode", awbmode_val)
            except Exception:
                pass

            # Apply controls after start (some pipelines only accept certain controls when running).
            self.picam2.start()
            self.cam_open = True

            if controls:
                ok = self._set_controls(controls)
                if ok:
                    self._log(logging.INFO, f"PiCam2:Controls set {controls}")

            self._log(logging.INFO, "PiCam2:Camera opened")
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
                except Exception:
                    pass
                try:
                    self.picam2.close()
                except Exception:
                    pass
        finally:
            self.picam2 = None
            self.cam_open = False
            self._last_metadata = None

    # ---------------------------------------------------------------------
    # Methods (need to fix name)
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

        if self._stream_policy == "maximize_fps":
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

    def get_metadata(self) -> dict:
        return self._metadata

    def get_control(self, name: str):
        return self._get_control(name)

    def set_controls(self, controls: dict) -> bool:
        return self._set_controls(controls)

    def convert(self, frame, to: str = "BGR888"):
        """Convert a captured frame to the requested OpenCV-friendly format.

        Supported targets: 'BGR888' (default), 'RGB888'. For Bayer RAW
        formats such as SRGGB10, SRGGB12, etc., this helper will demosaic
        and normalise higher bit-depth data down to 8-bit when an 8-bit
        packed target is requested. Returns input if conversion is
        unnecessary or fails.
        """
        if frame is None:
            return None
        to = (to or "BGR888").upper()
        code = self._cv2_color_conversion_code(to)
        if code is None:
            fmt = (self._format or "").upper()
            try:
                if fmt == "BGR888" and to == "RGB888" and getattr(frame, "ndim", 0) == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if fmt == "RGB888" and to == "BGR888" and getattr(frame, "ndim", 0) == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            except Exception:
                return frame
        try:
            out = cv2.cvtColor(frame, code)
        except Exception:
            return frame

        # If we converted from a high bit-depth RAW source (e.g. SRGGB10)
        # but the caller requested an 8-bit packed target ('BGR888' /
        # 'RGB888'), scale down to uint8 for display/processing.
        try:
            if (
                to in ("BGR888", "RGB888")
                and hasattr(out, "dtype")
                and np.issubdtype(out.dtype, np.integer)
                and out.dtype.itemsize > 1
            ):
                src_fmt = (self._format or "").upper()
                bit_depth = 16
                for cand in (16, 14, 12, 10):
                    if str(cand) in src_fmt:
                        bit_depth = cand
                        break
                shift = max(0, bit_depth - 8)
                if shift > 0:
                    out = (out >> shift).astype(np.uint8)
                else:
                    out = out.astype(np.uint8)
        except Exception:
            pass

        return out

    # ------------------------------------------------------------------
    # Optional logging helpers
    # ------------------------------------------------------------------

    def log_stream_options(self) -> None:
        """Log stream capabilities/options (on demand).

        - Output is sent to the log queue passed at construction time.
        - Only logs options relevant to the currently configured stream.
        """
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
        q = self.log
        if q is None:
            return
        try:
            if not q.full():
                q.put_nowait((int(level), str(msg)))
        except Exception:
            pass

    # Control and Format helpers
    # --------------------------

    def _get_control(self, name: str):
        """Get a single control/metadata value by name."""
        metadata = self._metadata
        if name in metadata:
            return metadata.get(name)
        else:
            if self.picam2 is None:
                # camera not open
                return None
            else:
                # update metadata and try again
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

    def _set_color_format(self, fmt: str | None):
        """Set and analyze the requested format string."""
        try:
            fmt_upper = (fmt or "").upper()  # type: ignore[arg-type]
        except Exception:
            fmt_upper = str(fmt).upper() if fmt is not None else ""

        self._format = fmt_upper if fmt_upper else None
        self._is_yuv420 = fmt_upper == "YUV420"
        non_alpha = set(self._supported_main_color_formats()) - {"XBGR8888", "XRGB8888"}
        self._needs_drop_alpha = bool(fmt_upper) and (not self._is_raw_format(fmt_upper)) and (fmt_upper not in non_alpha)
        if fmt_upper and self._needs_drop_alpha:
            self._log(logging.INFO, f"PiCam2:Dropping alpha channel for {fmt_upper} frames")

    def _list_sensor_modes(self):
        """List all sensor modes from Picamera2."""
        modes = []

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
                    modes.append({"format": fmt, "size": (w, h), "fps": fps, "area": int(w) * int(h)})
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

        desired_w, desired_h = (None, None)
        if desired_size and len(desired_size) == 2:
            desired_w, desired_h = int(desired_size[0]), int(desired_size[1])

        def score_fps_first(m):
            return (-float(m.get("fps", 0.0) or 0.0), -mode_area(m))

        def score_fov_first(m):
            return (-mode_area(m), -float(m.get("fps", 0.0) or 0.0))

        if self._stream_policy == "maximize_fov":
            modes.sort(key=score_fov_first)
        else:
            modes.sort(key=score_fps_first)

        if desired_w and desired_h:
            desired_area = desired_w * desired_h

            def add_area_penalty(m):
                return abs(mode_area(m) - desired_area)

            # Stable sort keeps the previous ordering as a tie-breaker.
            modes.sort(key=add_area_penalty)

        return modes[0]

    def _select_raw_sensor_mode(self, desired_size: tuple[int, int] | None = None):
        """Select the best raw sensor mode based on policy and desired size."""
        return self._select_sensor_mode(self._list_raw_sensor_modes(), desired_size=desired_size)

    def _select_main_sensor_mode(self, desired_size: tuple[int, int] | None = None):
        """Select the best main sensor mode based on policy and desired size."""
        modes = self._list_sensor_modes()
        return self._select_sensor_mode(modes, desired_size=desired_size)

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

    def _cv2_color_conversion_code(self, to: str):
        """Return OpenCV color conversion code from current format to target."""
        fmt = (self._format or "").upper()
        to = (to or "BGR888").upper()
        if not fmt:
            return None
        if (fmt == "BGR888" and to == "BGR888") or (fmt == "RGB888" and to == "RGB888"):
            return None
        if fmt == "BGR888" and to == "RGB888":
            return cv2.COLOR_BGR2RGB
        if fmt == "RGB888" and to == "BGR888":
            return cv2.COLOR_RGB2BGR
        # 32-bit RGB with dummy alpha channel (manual examples)
        # XBGR8888: pixels appear as [R, G, B, 255] in Python (RGBA order)
        # XRGB8888: pixels appear as [B, G, R, 255] in Python (BGRA order)
        if fmt == "XBGR8888":
            if to == "BGR888":
                return cv2.COLOR_RGBA2BGR
            if to == "RGB888":
                return cv2.COLOR_RGBA2RGB
        if fmt == "XRGB8888":
            if to == "BGR888":
                return cv2.COLOR_BGRA2BGR
            if to == "RGB888":
                return cv2.COLOR_BGRA2RGB
        if fmt == "YUV420":
            return cv2.COLOR_YUV2BGR_I420 if to == "BGR888" else cv2.COLOR_YUV2RGB_I420
        if fmt == "YUYV":
            return cv2.COLOR_YUV2BGR_YUY2 if to == "BGR888" else cv2.COLOR_YUV2RGB_YUY2
        if fmt.startswith("SRGGB"):
            return cv2.COLOR_BAYER_RG2BGR if to == "BGR888" else cv2.COLOR_BAYER_RG2RGB
        if fmt.startswith("SBGGR"):
            return cv2.COLOR_BAYER_BG2BGR if to == "BGR888" else cv2.COLOR_BAYER_BG2RGB
        if fmt.startswith("SGBRG"):
            return cv2.COLOR_BAYER_GB2BGR if to == "BGR888" else cv2.COLOR_BAYER_GB2RGB
        if fmt.startswith("SGRBG"):
            return cv2.COLOR_BAYER_GR2BGR if to == "BGR888" else cv2.COLOR_BAYER_GR2RGB
        return None


    @property
    def capturing(self)-> bool:
        return self._capturing

    @capturing.setter
    def capturing(self, value):
        self._capturing = bool(value)

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
        if value is None:
            return
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("size must be a (width, height) tuple")

        width, height = int(value[0]), int(value[1])
        if width <= 0 or height <= 0:
            raise ValueError("size must be positive")

        if self._capturing:
            self._log(logging.WARNING, "PiCam2:size change requires restart; stop capture first")
            return

        self._capture_width = width
        self._capture_height = height
        self._camera_res = (width, height)

        if self.cam_open:
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
        if value is None:
            return
        f = int(value)
        if f < 0 or f > 7:
            raise ValueError("flip must be in range 0..7")

        if self._capturing:
            self._log(logging.WARNING, "PiCam2:flip change requires restart; stop capture first")
            return

        self._flip_method = f
        try:
            self._configs["flip"] = f
        except Exception:
            pass

        if self.cam_open:
            self.close_cam()
            self.open_cam()

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
        fps = self._get_control("FrameRate")
        if fps is None:
            return float(self._framerate)
        return fps

    @fps.setter
    def fps(self, value):
        if value is None or value == -1:
            return
        self._framerate = float(value)
        self._set_controls({"FrameRate": float(value)})


__all__ = ["PiCamera2Core", "FrameBuffer"]
