###############################################################################
# Raspberry Pi CSI video capture (Picamera2 / libcamera)
#
# Uses the picamera2 library to interface with the Raspberry Pi Camera Module
#
# Mirrors the behavior of piCapture (legacy picamera) where practical:
# - Threaded capture into a bounded Queue
# - Same flip enum as cv2Capture (0..7)
# - Optional output resizing
#
# Urs Utzinger
# ChatGPT, OpenAI
#
# Changes:
# 2026 Optimizations
# 2025 Initial Release
###############################################################################

###############################################################################
# Imports
###############################################################################

from __future__ import annotations
from threading import Thread, Lock
from queue import Queue
import logging
import time
import cv2

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None  # type: ignore


class piCamera2Capture(Thread):
    """Threaded capture for Raspberry Pi CSI cameras using Picamera2."""

    def __init__(self,
        configs: dict,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        queue_size: int = 32,
    ):
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}

        # Normalize configs: derive main/raw keys from camera_res for consistent handling
        try:
            mode = str(self._configs.get("mode", "main")).lower()
            # Pick a base resolution: function arg > 'camera_res' > default
            base_res = None
            if res is not None:
                base_res = res
            else:
                base_res = self._configs.get("camera_res", (640, 480))
            if isinstance(base_res, (list, tuple)) and len(base_res) >= 2:
                base_res = (int(base_res[0]), int(base_res[1]))
            else:
                base_res = (640, 480)
            # Ensure camera_res present; derive others internally
            self._configs.setdefault("camera_res", base_res)
            # Formats defaults
            self._configs.setdefault("main_format", str(self._configs.get("main_format", self._configs.get("format", "BGR3"))))
            self._configs.setdefault("raw_format", str(self._configs.get("raw_format", self._configs.get("format", "SRGGB8"))))
        except Exception:
            # Best effort; downstream code has its own defaults
            pass

        self.camera_num = camera_num

        if exposure is not None:
            self._exposure = exposure
        else:
            self._exposure = self._configs.get("exposure", -1)

        if res is not None:
            self._camera_res = res
            # Keep configs consistent
            self._configs["camera_res"] = tuple(res)
        else:
            self._camera_res = self._configs.get("camera_res", (640, 480))

        self._capture_width = int(self._camera_res[0])
        self._capture_height = int(self._camera_res[1])
        # Main stream size derived from camera_res; raw size defaults handled by _raw_res

        self._output_res    = self._configs.get("output_res", (-1, -1))
        self._output_width  = int(self._output_res[0])
        self._output_height = int(self._output_res[1])

        self._framerate    = self._configs.get("fps", 30)
        self._flip_method  = self._configs.get("flip", 0)
        self._autoexposure = self._configs.get("autoexposure", -1)
        self._autowb       = self._configs.get("autowb", -1)
        # Preferred format name (Picamera2/libcamera), fallback to fourcc for legacy configs
        self._requested_format = str(self._configs.get("format", "")).upper()
        self._fourcc           = str(self._configs.get("fourcc", "")).upper()
        # Mode selection: 'main' (processed) or 'raw' (sensor window)
        self._mode = str(self._configs.get("mode", "main")).lower()
        # Policy for selecting sensor modes: 'default', 'maximize_fov', 'maximize_fps'
        self._stream_policy = str(self._configs.get("stream_policy", "default")).lower()
        # Separate main/raw format overrides (optional)
        self._main_format = str(self._configs.get("main_format", self._requested_format)).upper()
        self._raw_format     = str(self._configs.get("raw_format", self._requested_format)).upper()
        # Raw resolution override (optional), defaults to camera_res
        self._raw_res = self._configs.get("raw_res", (self._capture_width, self._capture_height))
        # Low-latency controls
        self._low_latency  = bool(self._configs.get("low_latency", False))
        # Optional explicit buffer_count override for Picamera2/libcamera
        self._buffer_count = self._configs.get("buffer_count", None)
        # Control whether to use libcamera hardware transform
        self._hw_transform = bool(self._configs.get("hw_transform", 1))
        # Resolved format and transform flags
        self._format: str | None = None
        self._stream_name: str = "main"
        self._needs_cpu_resize: bool = False
        self._needs_cpu_flip: bool = False
        self._is_yuv420: bool = False
        self._needs_drop_alpha: bool = False

        # Threading
        # Decide effective queue size: explicit buffersize wins; otherwise
        # low_latency prefers a size-1 queue (latest frame), else default.
        cfg_buffersize = self._configs.get("buffersize", None)
        if cfg_buffersize is not None:
            queue_size = int(cfg_buffersize)
        elif self._low_latency:
            queue_size = 1
        else:
            queue_size = int(queue_size)
        self.capture = Queue(maxsize=queue_size)
        self.log = Queue(maxsize=32)
        self.stopped = True
        self.cam_lock = Lock()

        # Runtime
        self.picam2 = None
        self.cam_open = False
        self._last_metadata: dict | None = None
        self.frame_time = 0.0
        self.measured_fps = 0.0

        self.open_cam()

    # ------------------------------------------------------------------
    # Friendly mappings for AE metering and AWB modes
    # ------------------------------------------------------------------

    # libcamera / Picamera2 AeMeteringMode:
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

    # libcamera / Picamera2 AwbMode:
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

    def stop(self):
        """Stop the thread."""
        self.stopped = True

    def start(self):
        """Start the capture thread."""
        self.stopped = False
        super().start()

    def run(self):
        """Thread entrypoint."""
        if not self.cam_open:
            return

        last_time = time.perf_counter()
        num_frames = 0

        try:
            while not self.stopped:
                current_time = time.perf_counter()

                if self.picam2 is None:
                    time.sleep(0.01)
                    continue

                try:
                    with self.cam_lock:
                        img = self.picam2.capture_array(self._stream_name)
                        self._last_metadata = self._capture_metadata_nolock()
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.WARNING, f"PiCam2:Capture failed: {exc}"))
                    time.sleep(0.005)
                    continue

                if img is None:
                    time.sleep(0.005)
                    continue

                num_frames += 1
                self.frame_time = int(current_time * 1000)

                img_proc = img

                # If the configured format yields 4 channels, drop alpha.
                if self._needs_drop_alpha:
                    try:
                        if getattr(img_proc, "ndim", 0) == 3 and img_proc.shape[2] == 4:
                            img_proc = img_proc[:, :, :3]
                    except Exception:
                        pass

                # Resize only when software resize is required (raw streams with output_res)
                if self._needs_cpu_resize:
                    img_proc = cv2.resize(img_proc, self._output_res)

                # Apply flip/rotation if requested (same enum as cv2Capture)
                if self._needs_cpu_flip:
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

                # Queue handling: low_latency prefers latest-frame semantics.
                if self._low_latency and self.capture.maxsize == 1:
                    # Overwrite any stale frame so consumer always sees newest.
                    try:
                        if not self.capture.empty():
                            _ = self.capture.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.capture.put_nowait((self.frame_time, img_proc))
                    except Exception:
                        pass
                else:
                    if not self.capture.full():
                        self.capture.put_nowait((self.frame_time, img_proc))
                    else:
                        if not self.log.full():
                            self.log.put_nowait((logging.WARNING, "PiCam2:Capture Queue is full!"))


                # FPS calculation
                if (current_time - last_time) >= 5.0:
                    self.measured_fps = num_frames / 5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:FPS:{self.measured_fps}"))
                    last_time = current_time
                    num_frames = 0
        finally:
            self.close_cam()

    def open_cam(self):
        if Picamera2 is None:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "PiCam2:picamera2 is not installed"))
            self.cam_open = False
            return

        try:
            self.picam2 = Picamera2(camera_num=self.camera_num)
            # Resolve requested format and stream
            req = (self._requested_format or self._fourcc or "").upper()
            # Decide initial stream based on mode
            if self._mode == "raw":
                self._stream_name = "raw"
                # Validate raw format/size against sensor modes
                rf = self._raw_format if self._is_raw_format(self._raw_format) else (req if self._is_raw_format(req) else "SRGGB8")
                try:
                    raw_w, raw_h = int(self._raw_res[0]), int(self._raw_res[1])
                    raw_size = (raw_w, raw_h)
                except Exception:
                    raw_size = (self._capture_width, self._capture_height)
                # First, validate against available raw modes
                orig_rf, orig_raw_size = rf, raw_size
                rf, raw_size = self._validate_raw_selection(rf, raw_size)
                # Then, let the policy helper pick a concrete sensor mode (if possible)
                selected_raw = self._select_raw_sensor_mode(desired_size=raw_size)
                if selected_raw is not None:
                    raw_size = selected_raw["size"]
                    rf = selected_raw["format"]
                    # For RAW, track the actual capture size for downstream logic/logging
                    try:
                        self._capture_width, self._capture_height = int(raw_size[0]), int(raw_size[1])
                    except Exception:
                        pass
                    if not self.log.full():
                        req_str = f"{orig_rf}@{orig_raw_size[0]}x{orig_raw_size[1]}" if isinstance(orig_raw_size, tuple) else str(orig_rf)
                        sel_str = f"{rf}@{raw_size[0]}x{raw_size[1]}"
                        self.log.put_nowait((
                            logging.INFO,
                            f"PiCam2:RAW sensor selection policy={self._stream_policy} requested={req_str} selected={sel_str} fps~{selected_raw['fps']}"
                        ))
                picam_format = rf
            else:
                self._stream_name = "main"
                # Prefer explicit main_format, then requested
                pfmt = self._main_format or req
                mapped = self._map_format(pfmt) or "BGR888"
                # Enforce supported main formats
                if mapped not in self.get_supported_main_formats():
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:Main format {mapped} not ideal; using BGR888. Supported main formats: {', '.join(self.get_supported_main_formats())}"))
                        self.log.put_nowait((logging.INFO, "PiCam2:For raw formats and resolutions, run examples/list_Picamera2Properties.py"))
                    mapped = "BGR888"
                picam_format = mapped
            self._set_format(picam_format)
            picam_format = self._format or picam_format
            
            # Choose main stream size: honor output_res when provided (avoids CPU resize).
            # Otherwise use camera_res for main stream.
            if (self._output_width > 0) and (self._output_height > 0):
                main_size = (self._output_width, self._output_height)
            else:
                main_size = (self._capture_width, self._capture_height)

            # For Main Stream, compute a preferred sensor mode (for logging/policy only).
            selected_main = None
            if self._stream_name == "main":
                try:
                    selected_main = self._select_main_sensor_mode(desired_size=main_size)
                    if selected_main is not None and not self.log.full():
                        fmt = selected_main["format"]
                        (sw, sh) = selected_main["size"]
                        fps = selected_main["fps"]
                        req_str = f"{main_size[0]}x{main_size[1]}"
                        sel_str = f"{fmt}@{sw}x{sh}"
                        self.log.put_nowait((
                            logging.INFO,
                            f"PiCam2:MAIN sensor selection policy={self._stream_policy} requested={req_str} selected={sel_str} fps~{fps}"
                        ))
                except Exception:
                    pass
            # Raw stream size validated above when in raw mode; otherwise derive from configs
            if self._stream_name != "raw":
                try:
                    raw_w, raw_h = int(self._raw_res[0]), int(self._raw_res[1])
                    raw_size = (raw_w, raw_h)
                except Exception:
                    raw_size = (self._capture_width, self._capture_height)

            # Try hardware transform via libcamera Transform (only if needed)
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
                    picam_kwargs = dict(raw={"size": raw_size, "format": picam_format},
                                  controls={"FrameRate": self._framerate})
                else:
                    picam_kwargs = dict(main={"size": main_size, "format": picam_format},
                                  controls={"FrameRate": self._framerate})
                # Optional low-latency: prefer small buffer_count when requested
                # unless user explicitly set buffer_count in configs.
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
                # Fallbacks
                if self._stream_name == "raw":
                    # try a safer raw format fallback
                    for fmt in ("SRGGB8", "SRGGB10_CSI2P"):
                        try:
                            picam_kwargs = dict(raw={"size": raw_size, "format": fmt},
                                          controls={"FrameRate": self._framerate})
                            if transform is not None:
                               picam_kwargs["transform"] = transform
                            config = self.picam2.create_video_configuration(**picam_kwargs)
                            self._set_format(fmt)
                            break
                        except Exception:
                            continue
                if config is None:
                    for fmt in ("BGR888", "RGB888"):
                        try:
                            picam_kwargs = dict(main={"size": main_size, "format": fmt},
                                          controls={"FrameRate": self._framerate})
                            if transform is not None:
                                picam_kwargs["transform"] = transform
                            config = self.picam2.create_video_configuration(**picam_kwargs)
                            self._set_format(fmt)
                            self._stream_name = "main"
                            break
                        except Exception:
                            continue

            self.picam2.configure(config)
            
            # Raw streams need software resize when an output size was requested
            self._needs_cpu_resize = (self._stream_name == "raw") and (self._output_width > 0) and (self._output_height > 0) and (not self._is_yuv420)

            # Apply initial controls
            controls = {}

            # Prefer full sensor field-of-view for processed streams only (not raw)
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
            autoexp  = self._autoexposure
            autowb   = self._autowb

            manual_requested = exposure is not None and exposure > 0
            if manual_requested:
                controls["AeEnable"] = False
                controls["ExposureTime"] = int(exposure)
            else:
                # Only touch AE when explicitly requested; ensure AeEnable is True/False
                if autoexp is None or autoexp == -1:
                    pass
                elif autoexp > 0:
                    controls["AeEnable"] = True
                else:
                    controls["AeEnable"] = False

            # Default AE metering: CentreWeighted (0) when not otherwise set.
            # Accept integers or friendly strings (e.g. 'center', 'spot', 'matrix').
            try:
                cfg_meter = self._configs.get("aemeteringmode", 0)
                meter_val = self._parse_aemeteringmode(cfg_meter)
                controls.setdefault("AeMeteringMode", meter_val)
            except Exception:
                pass

            # Only touch AWB when explicitly requested; ensure AwbEnable is True/False
            if autowb is None or autowb == -1:
                pass
            else:
                controls["AwbEnable"] = bool(autowb)

            # Default AWB mode: Auto (0) when not otherwise set.
            # Accept integers or friendly strings (e.g. 'auto', 'daylight', 'cloudy').
            try:
                cfg_awbmode = self._configs.get("awbmode", 0)
                awbmode_val = self._parse_awbmode(cfg_awbmode)
                controls.setdefault("AwbMode", awbmode_val)
            except Exception:
                pass

            if controls:
                # Best effort: set before and after start to ensure they stick
                self._set_controls(controls)

            self.picam2.start()
            self.cam_open = True

            if controls:
                ok = self._set_controls(controls)
                if ok and not self.log.full():
                    self.log.put_nowait((logging.INFO, f"PiCam2:Controls set {controls}"))

            # Emit a note about potential FOV cropping in raw windowed modes
            try:
                props = getattr(self.picam2, "camera_properties", {})
                aw, ah = None, None
                if isinstance(props, dict):
                    pas = props.get("PixelArraySize")
                    if isinstance(pas, (list, tuple)) and len(pas) == 2:
                        aw, ah = int(pas[0]), int(pas[1])
                if self._stream_name == "raw" and aw and ah:
                    rw, rh = self._capture_width, self._capture_height
                    if rw < aw and rh < ah:
                        if not self.log.full():
                            self.log.put_nowait((
                                logging.INFO,
                                f"PiCam2:RAW mode {rw}x{rh} is a sensor window (cropped FOV vs {aw}x{ah})."
                            ))
            except Exception:
                pass

            if not self.log.full():
                self.log.put_nowait((logging.INFO, "PiCam2:Camera opened"))
                # Informative summary
                if self._stream_name == "main":
                    self.log.put_nowait((logging.INFO, f"PiCam2:Main Stream mode {main_size[0]}x{main_size[1]} format={self._format}. Supported main formats: {', '.join(self.get_supported_main_formats())}"))
                    self.log.put_nowait((logging.INFO, "PiCam2:Main Stream can scale to arbitrary resolutions; non-native aspect ratios may crop. For raw modes list, run examples/list_Picamera2Properties.py."))
                    # List suggested Main Stream options (sensor-mode based)
                    try:
                        main_opts = self.get_suggested_main_options()
                        if main_opts:
                            self.log.put_nowait((logging.INFO, "PiCam2:Suggested Main Stream options (camera_res/output_res, max_fps, full_fov):"))
                            for opt in main_opts:
                                if self.log.full():
                                    break
                                cr = opt.get("camera_res", (0, 0))
                                mr = opt.get("output_res", cr)
                                fmt = opt.get("main_format", "BGR888")
                                fps = float(opt.get("max_fps", 0.0) or 0.0)
                                full = bool(opt.get("full_fov", False))
                                self.log.put_nowait((
                                    logging.INFO,
                                    f"PiCam2:  {cr[0]}x{cr[1]} -> {mr[0]}x{mr[1]} fmt={fmt} max_fps~{fps:.1f} full_fov={full}"
                                ))
                    except Exception:
                        pass
                else:
                    # RAW stream: list full RAW sensor options from get_supported_raw_options()
                    try:
                        modes = self.get_supported_raw_options()
                        if modes:
                            self.log.put_nowait((logging.INFO, f"PiCam2:Raw Stream {raw_size[0]}x{raw_size[1]} format={self._format}. Available RAW sensor modes (fmt@WxH~fps):"))
                            for m in modes:
                                if self.log.full():
                                    break
                                fmt = m.get('format')
                                size = m.get('size', (0, 0))
                                fps = float(m.get('fps', 0.0) or 0.0)
                                self.log.put_nowait((
                                    logging.INFO,
                                    f"PiCam2:  {fmt}@{size[0]}x{size[1]}~{fps:.1f}"
                                ))
                            self.log.put_nowait((logging.INFO, "PiCam2:Raw Stream resolutions/formats must match sensor modes. See examples/list_Picamera2Properties.py for details."))
                    except Exception:
                        pass

        except Exception as exc:
            self.picam2 = None
            self.cam_open = False
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"PiCam2:Failed to open camera: {exc}"))

    def close_cam(self):
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
    # Control and Format helpers
    # ---------------------------------------------------------------------

    def _capture_metadata_nolock(self) -> dict:
        """Return latest metadata from Picamera2.

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

    def _get_metadata(self) -> dict:
        if self._last_metadata is not None:
            return self._last_metadata
        if self.picam2 is None:
            return {}
        with self.cam_lock:
            self._last_metadata = self._capture_metadata_nolock()
            return self._last_metadata

    def _get_control(self, name: str):
        metadata = self._get_metadata()
        if name in metadata:
            return metadata.get(name)
        # Some controls may not show up in metadata; try reading again.
        if self.picam2 is None:
            return None
        with self.cam_lock:
            self._last_metadata = self._capture_metadata_nolock()
            return (self._last_metadata or {}).get(name)

    def _set_controls(self, controls: dict) -> bool:
        if self.picam2 is None:
            return False
        if not controls:
            return True
        try:
            with self.cam_lock:
                self.picam2.set_controls(controls)
            return True
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, f"PiCam2:Failed to set controls {controls}: {exc}"))
            return False

    # ------------------------------------------------------------------
    # Friendly parsers for AE metering and AWB mode
    # ------------------------------------------------------------------

    def _parse_aemeteringmode(self, value) -> int:
        """Parse AE metering from int or friendly string.

        Accepts integers (0/1/2) or strings like 'center', 'spot', 'matrix'.
        Defaults to 0 (CentreWeighted) on error.
        """
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
        """Parse AWB mode from int or friendly string.

        Accepts integers (0..5) or strings like 'auto', 'daylight', 'cloudy'.
        Defaults to 0 (Auto) on error.
        """
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

    def _map_fourcc(self, fourcc: str) -> str | None:
        """Map common FOURCC strings to Picamera2/libcamera format names."""
        if not fourcc:
            return None
        table = {
            "BGR3": "BGR888",
            "RGB3": "RGB888",
            "MJPG": "MJPEG",
            "YUY2": "YUYV",
            "YUYV": "YUYV",
            "YUV420": "YUV420",
            "YU12": "YUV420",  # planar 4:2:0
            "I420": "YUV420",  # planar 4:2:0
        }
        return table.get(fourcc)

    def _map_format(self, fmt: str) -> str | None:
        """Map requested format to libcamera name; pass through raw names."""
        if not fmt:
            return None
        if self._is_raw_format(fmt):
            return fmt
        # accept common names
        return self._map_fourcc(fmt) or (
            fmt if fmt in ("BGR888", "RGB888", "YUYV", "MJPEG", "YUV420") else None
        )

    def _is_raw_format(self, fmt: str) -> bool:
        if not fmt:
            return False
        fmt = fmt.upper()
        return fmt.startswith("SRGGB") or fmt.startswith("SBGGR") or fmt.startswith("SGBRG") or fmt.startswith("SGRBG")

    def _set_format(self, fmt: str | None):
        fmt_upper = (fmt or "").upper()
        self._format = fmt_upper if fmt_upper else None
        self._is_yuv420 = fmt_upper == "YUV420"
        non_alpha = {"BGR888", "RGB888", "YUV420", "YUYV", "MJPEG"}
        self._needs_drop_alpha = bool(fmt_upper) and (not self._is_raw_format(fmt_upper)) and (fmt_upper not in non_alpha)
        log_queue = getattr(self, "log", None)
        if fmt_upper and self._needs_drop_alpha and log_queue is not None:
            try:
                if not log_queue.full():
                    log_queue.put_nowait((logging.INFO, f"PiCam2:Dropping alpha channel for {fmt_upper} frames"))
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Supported options helpers
    # ---------------------------------------------------------------------

    def get_supported_main_formats(self):
        """Return main formats we support and can convert efficiently."""
        return ["BGR888", "RGB888", "YUV420", "YUYV"]

    def get_supported_raw_options(self):
        """Return available RAW (Bayer) sensor modes.

        RAW modes are Bayer-only formats whose names start with one of:
        SRGGB, SBGGR, SGBRG, SGRBG. YUV420 and other processed formats are
        *not* reported here.

        Returns a list of dicts with at least:
            {"format": str, "size": (w, h), "fps": float, "area": int}

        Older callers that only read "format" and "size" remain compatible.
        """
        modes = []
        try:
            all_modes = self._list_sensor_modes()
            for m in all_modes:
                try:
                    fmt = m.get("format")
                    size = m.get("size")
                    if not fmt or not size or not self._is_raw_format(fmt):
                        continue
                    modes.append({
                        "format": fmt,
                        "size": (int(size[0]), int(size[1])),
                        "fps": float(m.get("fps", 0.0) or 0.0),
                        "area": int(m.get("area", int(size[0]) * int(size[1])))
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return modes

    def _list_sensor_modes(self):
        """Internal: normalize Picamera2 sensor_modes to a common dict structure.

        Returns list of dicts: {"format", "size", "fps", "area"}.
        """
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
                    fps = m.get("fps", m.get("max_fps", 0))  # field name varies between builds
                    try:
                        fps = float(fps) if fps is not None else 0.0
                    except Exception:
                        fps = 0.0
                    modes.append({
                        "format": fmt,
                        "size": (w, h),
                        "fps": fps,
                        "area": w * h,
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return modes

    def _select_raw_sensor_mode(self, desired_size: tuple[int, int] | None = None):
        """Choose a RAW (Bayer) sensor mode.

        RAW mode only uses Bayer formats (SRGGB*/SBGGR*/SGBRG*/SGRBG*).

        Policy:
        - Default and 'maximize_fps': prefer highest FPS, then larger area.
        - 'maximize_fov': prefer largest area, then FPS.
        - desired_size is used as a soft tie-breaker (closest area).
        """
        modes = [m for m in self._list_sensor_modes() if self._is_raw_format(m["format"])]
        if not modes:
            return None

        desired_w, desired_h = (None, None)
        if desired_size and len(desired_size) == 2:
            desired_w, desired_h = int(desired_size[0]), int(desired_size[1])

        def score_fps_first(m):
            # Higher FPS first, then larger area
            return (-m["fps"], -m["area"])

        def score_fov_first(m):
            # Larger area first, then FPS
            return (-m["area"], -m["fps"])

        if self._stream_policy == "maximize_fov":
            modes.sort(key=score_fov_first)
        else:
            # default and 'maximize_fps': FPS first, then FOV
            modes.sort(key=score_fps_first)

        # Optional tie-breaker: closest area to desired
        if desired_w and desired_h:
            desired_area = desired_w * desired_h
            def add_area_penalty(m):
                return abs(m["area"] - desired_area)
            # stable sort: keep FPS/FOV ordering, refine by area difference
            modes.sort(key=add_area_penalty)

        return modes[0]

    def _select_main_sensor_mode(self, desired_size: tuple[int, int] | None = None):
        """Choose a sensor mode for Main Stream.

        Policy:
        - Prefer largest FOV (area) first.
        - Second priority: highest FPS.
        - desired_size is a soft tie-breaker towards requested resolution.
        - stream_policy == 'maximize_fps' inverts (FPS first, FOV second).
        """
        modes = self._list_sensor_modes()
        if not modes:
            return None

        desired_w, desired_h = (None, None)
        if desired_size and len(desired_size) == 2:
            desired_w, desired_h = int(desired_size[0]), int(desired_size[1])

        def score_fov_first(m):
            return (-m["area"], -m["fps"])

        def score_fps_first(m):
            return (-m["fps"], -m["area"])

        if self._stream_policy == "maximize_fps":
            modes.sort(key=score_fps_first)
        else:
            # 'default' and 'maximize_fov' behave the same initially
            modes.sort(key=score_fov_first)

        if desired_w and desired_h:
            desired_area = desired_w * desired_h
            def add_area_penalty(m):
                return abs(m["area"] - desired_area)
            modes.sort(key=add_area_penalty)

        return modes[0]

    def get_suggested_main_options(self, max_options: int = 8):
        """Return suggested Main Stream configurations for this camera.

        Each entry is a dict describing a "good" main-stream choice based on
        available sensor modes and this instance's policy:

            {
                "camera_res": (w, h),   # suggested camera_res
                "output_res": (w, h),   # suggested output_res
                "main_format": str,     # one of get_supported_main_formats()
                "max_fps": float,       # approximate max fps for this mode
                "full_fov": bool        # True if near-maximum sensor area
            }

        This helper is read-only and does not reconfigure the camera.
        """
        suggestions = []
        if self.picam2 is None:
            return suggestions

        modes = self._list_sensor_modes()
        if not modes:
            return suggestions

        # Determine what counts as full FOV (within 95% of max area)
        max_area = max(m["area"] for m in modes)
        full_fov_threshold = 0.95 * max_area

        # Deduplicate by size while preserving order from policy sort
        seen_sizes = set()
        # Sort using main sensor-mode selection policy
        preferred = self._select_main_sensor_mode(desired_size=self._camera_res)
        if preferred is not None:
            # Put preferred mode first if not already
            # (we still iterate over all modes below)
            pass

        # Sort modes FOV-first by default, or FPS-first if policy demands it
        if self._stream_policy == "maximize_fps":
            modes.sort(key=lambda m: (-m["fps"], -m["area"]))
        else:
            modes.sort(key=lambda m: (-m["area"], -m["fps"]))

        for m in modes:
            size = m["size"]
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                continue
            w, h = int(size[0]), int(size[1])
            if (w, h) in seen_sizes:
                continue
            seen_sizes.add((w, h))

            full_fov = m["area"] >= full_fov_threshold
            max_fps = float(m.get("fps", 0.0) or 0.0)

            suggestions.append({
                "camera_res": (w, h),
                "output_res": (w, h),
                "main_format": "BGR888",
                "max_fps": max_fps,
                "full_fov": bool(full_fov),
            })

            if len(suggestions) >= max_options:
                break

        return suggestions

    def _validate_raw_selection(self, desired_fmt: str, desired_size: tuple[int, int]):
        """Validate raw format and size against sensor modes, applying sensible fallbacks.

        Returns (fmt, size) potentially adjusted to a supported mode.
        Logs guidance if fallbacks are applied.
        """
        modes = self.get_supported_raw_options()
        if not modes:
            # No modes enumerated; keep requested, but warn.
            if not self.log.full():
                self.log.put_nowait((logging.INFO, "PiCam2:Raw modes unavailable; run examples/list_Picamera2Properties.py to list sensor modes."))
            return (desired_fmt, desired_size)

        desired_fmt = (desired_fmt or "").upper()
        sizes_for_fmt = [m["size"] for m in modes if m["format"].upper() == desired_fmt]
        if sizes_for_fmt:
            # If requested size is supported for the format, keep; else take first size for that format
            if tuple(desired_size) in sizes_for_fmt:
                return (desired_fmt, tuple(desired_size))
            sel_size = sizes_for_fmt[0]
            if not self.log.full():
                self.log.put_nowait((logging.INFO, f"PiCam2:Requested raw size {desired_size} not in {desired_fmt}; using {sel_size}."))
            return (desired_fmt, sel_size)

        # Requested format not available; pick the first available mode
        sel_fmt = modes[0]["format"]
        sel_size = modes[0]["size"]
        if not self.log.full():
            avail = ", ".join([f"{m['format']}@{m['size'][0]}x{m['size'][1]}" for m in modes])
            self.log.put_nowait((logging.INFO, f"PiCam2:Requested raw format {desired_fmt} not available; using {sel_fmt}@{sel_size[0]}x{sel_size[1]}"))
            self.log.put_nowait((logging.INFO, f"PiCam2:Available raw sensor modes: {avail}. For a full list run examples/list_Picamera2Properties.py"))
        return (sel_fmt, sel_size)

    # ---------------------------------------------------------------------
    # Public conversion helper
    # ---------------------------------------------------------------------

    def convert(self, frame, to: str = 'BGR888'):
        """Convert a captured frame to the requested OpenCV-friendly format.

        Supported targets: 'BGR888' (default), 'RGB888'. Returns input if
        conversion is unnecessary or fails.
        """
        if frame is None:
            return None
        to = (to or 'BGR888').upper()
        code = self._conversion_code(to)
        if code is None:
            # Already in target? Handle simple channel swap if needed.
            fmt = (self._format or '').upper()
            try:
                if fmt == 'BGR888' and to == 'RGB888' and getattr(frame, 'ndim', 0) == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if fmt == 'RGB888' and to == 'BGR888' and getattr(frame, 'ndim', 0) == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            except Exception:
                return frame
        try:
            return cv2.cvtColor(frame, code)
        except Exception:
            return frame

    def _conversion_code(self, to: str):
        fmt = (self._format or '').upper()
        to = (to or 'BGR888').upper()
        if not fmt:
            return None
        # No conversion needed when formats match
        if (fmt == 'BGR888' and to == 'BGR888') or (fmt == 'RGB888' and to == 'RGB888'):
            return None
        # Packed RGB/BGR swap
        if fmt == 'BGR888' and to == 'RGB888':
            return cv2.COLOR_BGR2RGB
        if fmt == 'RGB888' and to == 'BGR888':
            return cv2.COLOR_RGB2BGR
        # YUV planar and packed to BGR/RGB
        if fmt == 'YUV420':
            return cv2.COLOR_YUV2BGR_I420 if to == 'BGR888' else cv2.COLOR_YUV2RGB_I420
        if fmt == 'YUYV':
            return cv2.COLOR_YUV2BGR_YUY2 if to == 'BGR888' else cv2.COLOR_YUV2RGB_YUY2
        # Raw Bayer patterns demosaic
        if fmt.startswith('SRGGB'):
            return cv2.COLOR_BAYER_RG2BGR if to == 'BGR888' else cv2.COLOR_BAYER_RG2RGB
        if fmt.startswith('SBGGR'):
            return cv2.COLOR_BAYER_BG2BGR if to == 'BGR888' else cv2.COLOR_BAYER_BG2RGB
        if fmt.startswith('SGBRG'):
            return cv2.COLOR_BAYER_GB2BGR if to == 'BGR888' else cv2.COLOR_BAYER_GB2RGB
        if fmt.startswith('SGRBG'):
            return cv2.COLOR_BAYER_GR2BGR if to == 'BGR888' else cv2.COLOR_BAYER_GR2RGB
        return None

    # ---------------------------------------------------------------------
    # Flip options helpers
    # ---------------------------------------------------------------------

    def _flip_to_transform(self, flip: int, Transform):
        """Map cv2 flip enum to libcamera Transform."""
        try:
            t = Transform()
        except Exception:
            return None
        if flip == 0:
            return t
        if flip == 1:   # ccw 90
            t = Transform(hflip=0, vflip=0, rotation=270)
        elif flip == 2: # 180
            t = Transform(hflip=1, vflip=1)
        elif flip == 3: # cw 90
            t = Transform(hflip=0, vflip=0, rotation=90)
        elif flip == 4: # horizontal
            t = Transform(hflip=1, vflip=0)
        elif flip == 6: # vertical
            t = Transform(hflip=0, vflip=1)
        elif flip == 5: # upright diagonal (ccw90 + hflip)
            t = Transform(hflip=1, vflip=0, rotation=270)
        elif flip == 7: # upper-left diagonal (transpose)
            t = Transform(hflip=1, vflip=0, rotation=90)
        return t
        
    # ---------------------------------------------------------------------
    # Camera size/res helpers
    # ---------------------------------------------------------------------

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

        # Reconfiguration while the capture thread is running is risky.
        if not self.stopped:
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, "PiCam2:size change requires restart; stop() first"))
            return

        self._capture_width = width
        self._capture_height = height
        self._camera_res = (width, height)

        # If camera is open, reconfigure it.
        if self.cam_open:
            self.close_cam()
            self.open_cam()

    # ---------------------------------------------------------------------
    # Picamera2/libcamera Getters & Setters
    # Camera Controls
    # ---------------------------------------------------------------------

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

    # Auto Exposure

    @property
    def autoexposure(self):
        # Match cv2Capture semantics: -1 unknown, else 0/1.
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
        """AE metering mode. Default to CentreWeighted (0) if unknown."""
        val = self._get_control("AeMeteringMode")
        if val is None:
            return 0
        return int(val)
    @aemeteringmode.setter
    def aemeteringmode(self, value):
        """Set AE metering mode.

        Accepts either an int (0=CentreWeighted, 1=Spot, 2=Matrix)
        or a friendly string ('center', 'spot', 'matrix'). None/-1 -> 0.
        """
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

    # Auto White Balance

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
        """AWB mode. Default to Auto (0) if unknown."""
        val = self._get_control("AwbMode")
        if val is None:
            return 0
        return int(val)
    @awbmode.setter
    def awbmode(self, value):
        """Set AWB mode.

        Accepts either an int (0=Auto, 1=Tungsten, 2=Fluorescent,
        3=Indoor, 4=Daylight, 5=Cloudy) or a friendly string
        ('auto', 'tungsten', 'fluorescent', 'indoor', 'daylight', 'cloudy').
        None/-1 -> 0 (Auto).
        """
        awb_val = self._parse_awbmode(value)
        self._set_controls({"AwbMode": awb_val})

    # Color

    @property
    def wbtemperature(self):
        # Best-effort mapping: libcamera uses ColourTemperature.
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

    # Autofocus

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

    # Brightness, Contrast, Saturation, Sharpness, Noise Reduction

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


###############################################################################
# Testing
###############################################################################
if __name__ == '__main__':

    configs = {
        'camera_res'      : (1280, 720),    # any amera: Camera width & height
        'exposure'        : 10000,          # any camera: -1,0 = auto, 1...max=frame interval in microseconds
        'autoexposure'    : 0,              # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
        'fps'             : 60,             # any camera: 1/10, 15, 30, 40, 90, 120 overlocked
        'output_res'      : (-1, -1),       # Output resolution 
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

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")

    logger.log(logging.DEBUG, "Starting PiCamera2 capture")

    camera = piCamera2Capture(configs, camera_num=0)
    camera.start()

    cv2.namedWindow("Pi Camera 2", cv2.WINDOW_AUTOSIZE)

    try:
        while cv2.getWindowProperty("Pi Camera 2", 0) >= 0:
            try:
                (frame_time, frame) = camera.capture.get(timeout=0.25)
                cv2.imshow("Pi Camera 2", frame)
            except Exception:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            try:
                (level, msg) = camera.log.get_nowait()
                logger.log(level, f"PiCam2:{msg}")
            except Exception:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        try:
            camera.join(timeout=2.0)
        except Exception:
            pass
        try:
            camera.close_cam()
        except Exception:
            pass
        cv2.destroyAllWindows()