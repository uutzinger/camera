###############################################################################
# Basler pylon camera core (PyPylon / GenICam)
#
# Shared core used by both non-Qt threaded capture and Qt wrappers.
#
# Urs Utzinger
# GPT-5.2
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: PylonCore
#
# This is a non-threaded, non-Qt core wrapper intended to be composed by:
# - the threaded capture wrapper `pylonCapture`
# - the Qt wrapper `pylonCaptureQt`
#
# Public methods:
# - open_cam()         : Open and configure camera, allocate frame buffer.
# - close_cam()        : Stop and close camera, release resources.
# - capture_array()    : tuple[np.ndarray | None, float | None]
# - trigger_software() : bool
# - get_control(name: str) -> Any
# - set_controls(controls: dict) -> bool
# - get_supported_main_color_formats() -> list[str]
# - get_supported_raw_color_formats() -> list[str]
# - get_supported_raw_options() -> list[dict]
# - get_supported_main_options() -> list[dict]
# - log_stream_options() -> None
#
# Convenience properties:
# - cam_open: bool
# - buffer: FrameBuffer
# - metadata: dict
# - width / height / resolution / size
# - exposure, autoexposure, fps
# - flip, stream_policy
# - gain, gain_auto
# - acquisition_mode, exposure_mode, acquisition_frame_rate_enable
# - gamma, gamma_enable, balance_white_auto
# - binning, offset, adc, pixel_format
# - trigin, trigout, ttlinv
# - convert_format
#
# Notes:
# - Availability of controls varies by Basler model/interface.
# - Unsupported controls are logged and ignored.
###############################################################################

from __future__ import annotations

from threading import Lock
from queue import Queue
import logging
import time
from typing import TYPE_CHECKING, Any

import cv2

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .framebuffer import FrameBuffer

try:
    from pypylon import pylon, genicam
except Exception:  # pragma: no cover
    pylon = None  # type: ignore
    genicam = None  # type: ignore


class PylonCore:
    """Core PyPylon wrapper."""

    _CONVERT_FORMATS = ("native", "Mono8", "Mono16", "BGR8", "RGB8")

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

        base_res = res if res is not None else self._configs.get("camera_res", (720, 540))
        if isinstance(base_res, (list, tuple)) and len(base_res) >= 2:
            self._camera_res = (int(base_res[0]), int(base_res[1]))
        else:
            self._camera_res = (720, 540)
        self._configs.setdefault("camera_res", self._camera_res)

        self._capture_width = int(self._camera_res[0])
        self._capture_height = int(self._camera_res[1])

        self._output_res = self._configs.get("output_res", (-1, -1))
        self._output_width = int(self._output_res[0])
        self._output_height = int(self._output_res[1])
        self._flip_method = int(self._configs.get("flip", 0))

        self._stream_policy = self._normalize_stream_policy(self._configs.get("stream_policy", "default"))

        self._framerate = float(self._configs.get("fps", 30) or 30)
        self._autoexposure = int(self._configs.get("autoexposure", -1))
        self._exposure = float(exposure if exposure is not None else self._configs.get("exposure", -1))

        self._cfg_exposure_present = (exposure is not None) or ("exposure" in self._configs)
        self._cfg_autoexposure_present = "autoexposure" in self._configs
        self._cfg_fps_present = "fps" in self._configs

        self._binning = self._configs.get("binning", (1, 1))
        self._offset = self._configs.get("offset", (0, 0))
        try:
            self._adc = int(self._configs.get("adc", -1))
        except Exception:
            self._adc = -1
        self._cfg_adc_present = "adc" in self._configs
        self._pixel_format_req = str(self._configs.get("pixel_format", "") or "")
        self._convert_format = str(self._configs.get("convert_format", "native") or "native")
        if self._convert_format not in self._CONVERT_FORMATS:
            self._convert_format = "native"

        self._trigout = int(self._configs.get("trigout", -1))
        self._ttlinv = bool(self._configs.get("ttlinv", False))
        self._trigin = int(self._configs.get("trigin", -2))  # -2 means leave camera default
        self._cfg_trigout_present = "trigout" in self._configs
        self._cfg_ttlinv_present = "ttlinv" in self._configs
        self._cfg_trigin_present = "trigin" in self._configs

        self._cfg_isp_enable = self._configs.get("isp_enable") if "isp_enable" in self._configs else None
        self._cfg_gain = self._configs.get("gain") if "gain" in self._configs else None
        self._cfg_gain_auto = self._configs.get("gain_auto") if "gain_auto" in self._configs else None
        self._cfg_acquisition_mode = (
            self._configs.get("acquisition_mode") if "acquisition_mode" in self._configs else None
        )
        self._cfg_exposure_mode = self._configs.get("exposure_mode") if "exposure_mode" in self._configs else None
        self._cfg_acquisition_frame_rate_enable = (
            self._configs.get("acquisition_frame_rate_enable")
            if "acquisition_frame_rate_enable" in self._configs
            else None
        )
        self._cfg_gamma = self._configs.get("gamma") if "gamma" in self._configs else None
        self._cfg_gamma_enable = self._configs.get("gamma_enable") if "gamma_enable" in self._configs else None
        self._cfg_balance_white_auto = (
            self._configs.get("balance_white_auto") if "balance_white_auto" in self._configs else None
        )

        self._capture_timeout_ms = int(self._configs.get("capture_timeout_ms", 100))
        if self._capture_timeout_ms <= 0:
            self._capture_timeout_ms = 100

        self._buffer_capacity = int(self._configs.get("buffersize", 32))
        if self._buffer_capacity < 1:
            self._buffer_capacity = 1
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        self._buffer: FrameBuffer | None = None

        self._cam_open = False
        self._metadata: dict[str, Any] = {}
        self.cam = None
        self.cam_lock = Lock()
        self._converter = None
        self._prefetched_frame: tuple[np.ndarray, float] | None = None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str) -> None:
        q = self.log
        if q is None:
            return
        try:
            if not q.full():
                q.put_nowait((int(level), str(message)))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Generic parameter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_stream_policy(policy: str | None) -> str:
        v = str(policy or "default").strip().lower()
        if v in ("max", "maxfps", "max_fps", "performance", "fast"):
            return "max_fps"
        if v in ("auto", "automatic"):
            return "auto"
        return "default"

    def _param(self, name: str):
        if not self._cam_open or self.cam is None:
            return None
        return getattr(self.cam, name, None)

    def _enum_choices(self, name: str) -> list[str]:
        p = self._param(name)
        if p is None:
            return []
        try:
            if hasattr(p, "GetSymbolics"):
                vals = list(p.GetSymbolics())
                return [str(v) for v in vals]
        except Exception:
            pass
        try:
            vals = list(getattr(p, "Symbolics", []))
            return [str(v) for v in vals]
        except Exception:
            pass
        # Fallback via node map
        if genicam is None or self.cam is None:
            return []
        try:
            node = self.cam.GetNodeMap().GetNode(name)
            if node is None:
                return []
            enum = genicam.CEnumerationPtr(node)
            if enum is None:
                return []
            out: list[str] = []
            for ent in enum.GetEntries():
                try:
                    if genicam.IsAvailable(ent) and genicam.IsReadable(ent):
                        out.append(str(ent.GetSymbolic()))
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _enum_set(self, name: str, value) -> bool:
        p = self._param(name)
        if p is None:
            return False
        target = str(value)
        # Exact
        try:
            p.SetValue(target)
            return True
        except Exception:
            pass
        # Case-insensitive fallback
        options = self._enum_choices(name)
        for opt in options:
            if opt.lower() == target.lower():
                try:
                    p.SetValue(opt)
                    return True
                except Exception:
                    return False
        return False

    def _enum_get(self, name: str, default: str = "") -> str:
        p = self._param(name)
        if p is None:
            return default
        try:
            return str(p.GetValue())
        except Exception:
            try:
                return str(p.Value)
            except Exception:
                return default

    def _set_number(self, name: str, value, *, as_int: bool = False) -> bool:
        p = self._param(name)
        if p is None:
            return False
        try:
            v = int(value) if as_int else float(value)
        except Exception:
            return False
        try:
            if hasattr(p, "GetMin"):
                lo = p.GetMin()
                if lo is not None and v < lo:
                    v = lo
            if hasattr(p, "GetMax"):
                hi = p.GetMax()
                if hi is not None and v > hi:
                    v = hi
        except Exception:
            pass
        try:
            p.SetValue(int(v) if as_int else float(v))
            return True
        except Exception:
            return False

    def _get_number(self, name: str, default: float | int, *, as_int: bool = False):
        p = self._param(name)
        if p is None:
            return default
        try:
            v = p.GetValue()
            return int(v) if as_int else float(v)
        except Exception:
            try:
                v = p.Value
                return int(v) if as_int else float(v)
            except Exception:
                return default

    def _set_bool(self, name: str, value: bool) -> bool:
        p = self._param(name)
        if p is None:
            return False
        try:
            p.SetValue(bool(value))
            return True
        except Exception:
            try:
                p.Value = bool(value)
                return True
            except Exception:
                return False

    def _get_bool(self, name: str, default: bool = False) -> bool:
        p = self._param(name)
        if p is None:
            return bool(default)
        try:
            return bool(p.GetValue())
        except Exception:
            try:
                return bool(p.Value)
            except Exception:
                return bool(default)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def open_cam(self) -> bool:
        if self._cam_open:
            return True

        if pylon is None:
            self._log(logging.CRITICAL, "PyPylon:pypylon not installed")
            return False

        try:
            with self.cam_lock:
                factory = pylon.TlFactory.GetInstance()
                devices = factory.EnumerateDevices()
                if len(devices) <= 0:
                    self._log(logging.CRITICAL, "PyPylon:No cameras detected")
                    return False

                if self._camera_num < 0 or self._camera_num >= len(devices):
                    self._log(logging.WARNING, f"PyPylon:camera_num {self._camera_num} out of range; using 0")
                    self._camera_num = 0

                dev = devices[self._camera_num]
                self.cam = pylon.InstantCamera(factory.CreateDevice(dev))
                self.cam.Open()
                self._cam_open = True

                self._metadata = {
                    "model": self._safe_dev_info(dev, "GetModelName"),
                    "serial": self._safe_dev_info(dev, "GetSerialNumber"),
                    "vendor": self._safe_dev_info(dev, "GetVendorName"),
                    "device_class": self._safe_dev_info(dev, "GetDeviceClass"),
                }

                self._apply_stream_policy_baseline()
                self._apply_config_overrides()
                self._configure_converter()

                self._start_grabbing_locked()
                frame, ts_ms = self._retrieve_frame_locked()
                if frame is None:
                    shape, dtype = self._fallback_shape_dtype()
                    self._allocate_buffer(shape, dtype)
                    self._log(logging.WARNING, "PyPylon:No initial frame; allocated fallback buffer")
                else:
                    self._allocate_buffer(frame.shape, frame.dtype)
                    self._prefetched_frame = (frame, float(ts_ms if ts_ms is not None else time.perf_counter() * 1000.0))

            return True
        except Exception as exc:
            self._log(logging.CRITICAL, f"PyPylon:Failed to open camera: {exc}")
            self.close_cam()
            return False

    def close_cam(self) -> None:
        with self.cam_lock:
            cam = self.cam
            self.cam = None
            self._prefetched_frame = None
            self._converter = None
            self._buffer = None
            self._cam_open = False
            if cam is None:
                return
            try:
                if cam.IsGrabbing():
                    cam.StopGrabbing()
            except Exception:
                pass
            try:
                if cam.IsOpen():
                    cam.Close()
            except Exception:
                pass

    def _safe_dev_info(self, dev, getter: str) -> str | None:
        try:
            fn = getattr(dev, getter, None)
            if fn is None:
                return None
            return str(fn())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Configuration application
    # ------------------------------------------------------------------

    def _apply_stream_policy_baseline(self) -> None:
        if not self._cam_open:
            return

        policy = self._stream_policy
        if policy == "max_fps":
            self._apply_auto_mode("ExposureAuto", False)
            self._apply_auto_mode("GainAuto", False)
            self._apply_auto_mode("BalanceWhiteAuto", False)
            self._set_bool("AcquisitionFrameRateEnable", True)
            self._set_enum_best_effort("AcquisitionMode", "Continuous")
            self._set_enum_best_effort("ExposureMode", "Timed")
            self._set_number("Gain", 1.0)
            self._set_number("Gamma", 1.0)
            self._set_bool("GammaEnable", False)
            self._set_isp_enable_best_effort(False)
        elif policy == "auto":
            self._apply_auto_mode("ExposureAuto", True)
            self._apply_auto_mode("GainAuto", True)
            self._apply_auto_mode("BalanceWhiteAuto", True)
            self._set_enum_best_effort("AcquisitionMode", "Continuous")
            self._set_bool("GammaEnable", True)
            self._set_isp_enable_best_effort(True)

    def _apply_config_overrides(self) -> None:
        # Geometry first
        self._set_resolution_locked(self._capture_width, self._capture_height, regrab=False)

        # Main controls
        if self._cfg_acquisition_mode is not None:
            self.acquisition_mode = str(self._cfg_acquisition_mode)
        if self._cfg_exposure_mode is not None:
            self.exposure_mode = str(self._cfg_exposure_mode)
        if self._cfg_acquisition_frame_rate_enable is not None:
            self.acquisition_frame_rate_enable = bool(self._cfg_acquisition_frame_rate_enable)

        if self._cfg_autoexposure_present:
            self.autoexposure = int(self._autoexposure)
        if self._cfg_exposure_present:
            self.exposure = float(self._exposure)
        if self._cfg_fps_present:
            self.fps = float(self._framerate)

        if self._cfg_gain_auto is not None:
            self.gain_auto = self._cfg_gain_auto
        if self._cfg_gain is not None:
            self.gain = float(self._cfg_gain)

        if self._cfg_balance_white_auto is not None:
            self.balance_white_auto = self._cfg_balance_white_auto
        if self._cfg_gamma_enable is not None:
            self.gamma_enable = bool(self._cfg_gamma_enable)
        if self._cfg_gamma is not None:
            self.gamma = float(self._cfg_gamma)
        if self._cfg_isp_enable is not None:
            self.isp_enable = bool(self._cfg_isp_enable)

        # Format / geometry controls
        self.binning = self._binning
        self.offset = self._offset
        if self._cfg_adc_present:
            self.adc = self._adc
        if self._pixel_format_req:
            self.pixel_format = self._pixel_format_req
        self.convert_format = self._convert_format

        # Trigger/IO controls
        if self._cfg_ttlinv_present:
            self.ttlinv = self._ttlinv
        if self._cfg_trigin_present:
            self.trigin = self._trigin
        if self._cfg_trigout_present:
            self.trigout = self._trigout

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _start_grabbing_locked(self) -> None:
        if self.cam is None:
            return
        try:
            if not self.cam.IsGrabbing():
                self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            # Fallback if strategy enum is unavailable
            try:
                if not self.cam.IsGrabbing():
                    self.cam.StartGrabbing()
            except Exception as exc:
                self._log(logging.ERROR, f"PyPylon:StartGrabbing failed: {exc}")

    def _stop_grabbing_locked(self) -> None:
        if self.cam is None:
            return
        try:
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()
        except Exception:
            pass

    def _retrieve_frame_locked(self) -> tuple[np.ndarray | None, float | None]:
        cam = self.cam
        if cam is None or (not self._cam_open):
            return None, None

        self._start_grabbing_locked()

        result = None
        try:
            result = cam.RetrieveResult(self._capture_timeout_ms, pylon.TimeoutHandling_Return)
            if result is None:
                return None, None
            if not result.GrabSucceeded():
                return None, None

            ts_ms = self._timestamp_ms(result)
            frame = self._result_to_array(result)
            if frame is None:
                return None, None

            frame = self._apply_cpu_postprocessing(frame)
            return frame, ts_ms
        except Exception as exc:
            self._log(logging.WARNING, f"PyPylon:RetrieveResult failed: {exc}")
            return None, None
        finally:
            try:
                if result is not None:
                    result.Release()
            except Exception:
                pass

    def capture_array(self) -> tuple[np.ndarray | None, float | None]:
        if not self._cam_open:
            if not self.open_cam():
                return None, None

        if self._prefetched_frame is not None:
            frame, ts_ms = self._prefetched_frame
            self._prefetched_frame = None
            return frame, ts_ms

        with self.cam_lock:
            return self._retrieve_frame_locked()

    def _timestamp_ms(self, result) -> float:
        # PyPylon timestamp is camera-dependent; if unavailable, use monotonic host time.
        candidates = (
            "TimeStamp",
            "Timestamp",
            "GetTimeStamp",
            "GetTimestamp",
        )
        for attr in candidates:
            try:
                v = getattr(result, attr)
                if callable(v):
                    v = v()
                if v is None:
                    continue
                fv = float(v)
                # Most cameras report ns or us; treat large numbers as ns first.
                if fv > 1e12:
                    return fv * 1e-6
                if fv > 1e9:
                    return fv * 1e-3
                return fv
            except Exception:
                continue
        return time.perf_counter() * 1000.0

    def _result_to_array(self, result):
        try:
            if self._convert_format == "native":
                return result.Array
            self._configure_converter()
            if self._converter is None:
                return result.Array
            conv = self._converter.Convert(result)
            if conv is None:
                return result.Array
            return conv.GetArray()
        except Exception as exc:
            self._log(logging.WARNING, f"PyPylon:Format conversion failed: {exc}")
            try:
                return result.Array
            except Exception:
                return None

    def _configure_converter(self) -> None:
        if pylon is None:
            self._converter = None
            return
        if self._convert_format == "native":
            self._converter = None
            return

        converter = pylon.ImageFormatConverter()
        pixel_const = None
        if self._convert_format == "Mono8":
            pixel_const = getattr(pylon, "PixelType_Mono8", None)
        elif self._convert_format == "Mono16":
            pixel_const = getattr(pylon, "PixelType_Mono16", None)
        elif self._convert_format == "BGR8":
            pixel_const = getattr(pylon, "PixelType_BGR8packed", None)
        elif self._convert_format == "RGB8":
            pixel_const = getattr(pylon, "PixelType_RGB8packed", None)

        if pixel_const is None:
            self._log(logging.WARNING, f"PyPylon:convert_format {self._convert_format} unsupported; using native")
            self._convert_format = "native"
            self._converter = None
            return

        try:
            converter.OutputPixelFormat = pixel_const
            if hasattr(converter, "OutputBitAlignment"):
                align = getattr(pylon, "OutputBitAlignment_MsbAligned", None)
                if align is not None:
                    converter.OutputBitAlignment = align
            self._converter = converter
        except Exception as exc:
            self._converter = None
            self._log(logging.WARNING, f"PyPylon:Failed to configure converter: {exc}")

    def _apply_cpu_postprocessing(self, frame):
        out = frame
        # Resize if configured
        if self._output_width > 0 and self._output_height > 0:
            try:
                if out.shape[1] != self._output_width or out.shape[0] != self._output_height:
                    out = cv2.resize(out, (self._output_width, self._output_height), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass

        # Flip / rotate
        f = int(self._flip_method)
        if f == 0:
            return out
        try:
            if f == 1:
                return cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if f == 2:
                return cv2.rotate(out, cv2.ROTATE_180)
            if f == 3:
                return cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            if f == 4:
                return cv2.flip(out, 1)
            if f == 5:
                return cv2.transpose(out)
            if f == 6:
                return cv2.flip(out, 0)
            if f == 7:
                return cv2.flip(cv2.transpose(out), 1)
        except Exception:
            return out
        return out

    # ------------------------------------------------------------------
    # FrameBuffer management
    # ------------------------------------------------------------------

    def _fallback_shape_dtype(self) -> tuple[tuple[int, ...], np.dtype]:
        width = max(1, int(self._capture_width))
        height = max(1, int(self._capture_height))
        if self._output_width > 0 and self._output_height > 0:
            width = self._output_width
            height = self._output_height
        if self._convert_format in ("BGR8", "RGB8"):
            return (height, width, 3), np.dtype(np.uint8)
        if self._convert_format == "Mono16":
            return (height, width), np.dtype(np.uint16)
        return (height, width), np.dtype(np.uint8)

    def _allocate_buffer(self, frame_shape: tuple[int, ...], dtype) -> None:
        self._buffer = FrameBuffer(
            capacity=self._buffer_capacity,
            frame_shape=tuple(int(x) for x in frame_shape),
            dtype=np.dtype(dtype),
            overwrite=self._buffer_overwrite,
        )

    def _refresh_buffer_from_camera_locked(self) -> None:
        frame, ts_ms = self._retrieve_frame_locked()
        if frame is None:
            shape, dtype = self._fallback_shape_dtype()
            self._allocate_buffer(shape, dtype)
            self._prefetched_frame = None
            return
        self._allocate_buffer(frame.shape, frame.dtype)
        self._prefetched_frame = (frame, float(ts_ms if ts_ms is not None else time.perf_counter() * 1000.0))

    # ------------------------------------------------------------------
    # Trigger helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _line_name(line: int) -> str:
        if line == 0:
            return "Line1"
        if line > 0:
            return f"Line{line}"
        return ""

    def trigger_software(self) -> bool:
        if not self._cam_open or self.cam is None:
            return False
        with self.cam_lock:
            try:
                # Preferred API on InstantCamera
                if hasattr(self.cam, "WaitForFrameTriggerReady"):
                    try:
                        self.cam.WaitForFrameTriggerReady(self._capture_timeout_ms, pylon.TimeoutHandling_Return)
                    except Exception:
                        pass
                if hasattr(self.cam, "ExecuteSoftwareTrigger"):
                    self.cam.ExecuteSoftwareTrigger()
                    return True
            except Exception:
                pass
            # Fallback through node
            try:
                trig = self._param("TriggerSoftware")
                if trig is not None and hasattr(trig, "Execute"):
                    trig.Execute()
                    return True
            except Exception as exc:
                self._log(logging.WARNING, f"PyPylon:TriggerSoftware failed: {exc}")
        return False

    def _apply_trigger_input_locked(self) -> None:
        line = int(self._trigin)
        # -2: leave camera default untouched
        if line == -2:
            return
        if line < -1:
            self._set_enum_best_effort("TriggerMode", "Off")
            return
        self._set_enum_best_effort("TriggerSelector", "FrameStart")
        self._set_enum_best_effort("TriggerMode", "On")
        if line == -1:
            self._set_enum_best_effort("TriggerSource", "Software")
            return
        source = self._line_name(line)
        if not source:
            return
        if not self._enum_set("TriggerSource", source):
            self._log(logging.WARNING, f"PyPylon:TriggerSource {source} not available")
            return
        if self._ttlinv:
            self._set_enum_best_effort("TriggerActivation", "FallingEdge")
        else:
            self._set_enum_best_effort("TriggerActivation", "RisingEdge")

    def _apply_trigger_output_locked(self) -> None:
        line = int(self._trigout)
        if line < 0:
            return
        line_name = self._line_name(line)
        if not line_name:
            return
        if not self._enum_set("LineSelector", line_name):
            self._log(logging.WARNING, f"PyPylon:Line {line_name} not available for trigout")
            return
        self._set_enum_best_effort("LineMode", "Output")
        self._set_bool("LineInverter", bool(self._ttlinv))
        # ExposureActive is preferred for frame-exposure gating.
        if not self._enum_set("LineSource", "ExposureActive"):
            self._enum_set("LineSource", "FrameActive")

    # ------------------------------------------------------------------
    # Enum/auto helpers
    # ------------------------------------------------------------------

    def _set_enum_best_effort(self, name: str, value: str) -> bool:
        if self._enum_set(name, value):
            return True
        return False

    def _apply_auto_mode(self, name: str, enabled: bool) -> bool:
        if enabled:
            for mode in ("Continuous", "On", "Once"):
                if self._enum_set(name, mode):
                    return True
            return False
        for mode in ("Off", "Disabled"):
            if self._enum_set(name, mode):
                return True
        return False

    def _auto_mode_to_int(self, name: str, default: int = -1) -> int:
        val = self._enum_get(name, "")
        if not val:
            return int(default)
        v = val.lower()
        if v in ("off", "disabled"):
            return 0
        return 1

    # ------------------------------------------------------------------
    # Control properties
    # ------------------------------------------------------------------

    @property
    def cam_open(self) -> bool:
        return bool(self._cam_open)

    @property
    def buffer(self) -> FrameBuffer | None:
        return self._buffer

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    @property
    def width(self) -> int:
        if self._cam_open:
            self._capture_width = int(self._get_number("Width", self._capture_width, as_int=True))
        return int(self._capture_width)

    @width.setter
    def width(self, val) -> None:
        try:
            w = int(val)
        except Exception:
            return
        self._set_resolution_locked(w, self._capture_height, regrab=True)

    @property
    def height(self) -> int:
        if self._cam_open:
            self._capture_height = int(self._get_number("Height", self._capture_height, as_int=True))
        return int(self._capture_height)

    @height.setter
    def height(self, val) -> None:
        try:
            h = int(val)
        except Exception:
            return
        self._set_resolution_locked(self._capture_width, h, regrab=True)

    def _set_resolution_locked(self, width: int, height: int, *, regrab: bool) -> None:
        width = max(1, int(width))
        height = max(1, int(height))
        self._capture_width = width
        self._capture_height = height
        try:
            self._configs["camera_res"] = (width, height)
        except Exception:
            pass
        if not self._cam_open or self.cam is None:
            return
        with self.cam_lock:
            was_grabbing = self.cam.IsGrabbing()
            if was_grabbing:
                self._stop_grabbing_locked()
            self._set_number("Width", width, as_int=True)
            self._set_number("Height", height, as_int=True)
            if was_grabbing:
                self._start_grabbing_locked()
            if regrab:
                self._refresh_buffer_from_camera_locked()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @resolution.setter
    def resolution(self, val) -> None:
        self.size = val

    @property
    def size(self) -> tuple[int, int]:
        return self.resolution

    @size.setter
    def size(self, val) -> None:
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            return
        try:
            w = int(val[0])
            h = int(val[1])
        except Exception:
            return
        self._set_resolution_locked(w, h, regrab=True)

    @property
    def exposure(self) -> float:
        if self._cam_open:
            self._exposure = float(self._get_number("ExposureTime", self._exposure))
        return float(self._exposure)

    @exposure.setter
    def exposure(self, val) -> None:
        try:
            ev = float(val)
        except Exception:
            return
        self._exposure = ev
        try:
            self._configs["exposure"] = ev
        except Exception:
            pass
        if self._cam_open:
            if not self._set_number("ExposureTime", ev):
                self._log(logging.WARNING, "PyPylon:ExposureTime not available")

    @property
    def autoexposure(self) -> int:
        if self._cam_open:
            self._autoexposure = self._auto_mode_to_int("ExposureAuto", self._autoexposure)
        return int(self._autoexposure)

    @autoexposure.setter
    def autoexposure(self, val) -> None:
        try:
            iv = int(val)
        except Exception:
            return
        self._autoexposure = iv
        try:
            self._configs["autoexposure"] = iv
        except Exception:
            pass
        if not self._cam_open:
            return
        if iv < 0:
            return
        if not self._apply_auto_mode("ExposureAuto", iv > 0):
            self._log(logging.WARNING, "PyPylon:ExposureAuto not available")

    @property
    def fps(self) -> float:
        if self._cam_open:
            self._framerate = float(self._get_number("AcquisitionFrameRate", self._framerate))
        return float(self._framerate)

    @fps.setter
    def fps(self, val) -> None:
        try:
            fv = float(val)
        except Exception:
            return
        self._framerate = fv
        try:
            self._configs["fps"] = fv
        except Exception:
            pass
        if self._cam_open:
            ok = self._set_number("AcquisitionFrameRate", fv)
            if not ok:
                self._log(logging.WARNING, "PyPylon:AcquisitionFrameRate not available")

    @property
    def flip(self) -> int:
        return int(self._flip_method)

    @flip.setter
    def flip(self, val) -> None:
        try:
            f = int(val)
        except Exception:
            return
        self._flip_method = f
        try:
            self._configs["flip"] = f
        except Exception:
            pass
        if self._cam_open and self.cam is not None:
            with self.cam_lock:
                self._refresh_buffer_from_camera_locked()

    @property
    def stream_policy(self) -> str:
        return str(self._stream_policy)

    @stream_policy.setter
    def stream_policy(self, policy: str) -> None:
        self._stream_policy = self._normalize_stream_policy(policy)
        try:
            self._configs["stream_policy"] = self._stream_policy
        except Exception:
            pass
        if self._cam_open:
            self._apply_stream_policy_baseline()

    @property
    def isp_enable(self) -> bool:
        return bool(self._get_isp_enable_best_effort(default=bool(self._cfg_isp_enable) if self._cfg_isp_enable is not None else False))

    @isp_enable.setter
    def isp_enable(self, enabled: bool) -> None:
        self._cfg_isp_enable = bool(enabled)
        try:
            self._configs["isp_enable"] = bool(enabled)
        except Exception:
            pass
        if self._cam_open:
            self._set_isp_enable_best_effort(bool(enabled))

    def _set_isp_enable_best_effort(self, enabled: bool) -> None:
        # Basler models differ widely. Try a few known knobs and ignore if absent.
        if self._set_bool("ProcessedRawEnable", enabled):
            return
        # Some models use PgiMode enum.
        self._enum_set("PgiMode", "On" if enabled else "Off")

    def _get_isp_enable_best_effort(self, default: bool = False) -> bool:
        p = self._param("ProcessedRawEnable")
        if p is not None:
            return self._get_bool("ProcessedRawEnable", default)
        v = self._enum_get("PgiMode", "")
        if v:
            return v.lower() in ("on", "enabled", "true", "continuous")
        return bool(default)

    @property
    def gain(self) -> float:
        if self._cam_open:
            self._cfg_gain = float(self._get_number("Gain", self._cfg_gain if self._cfg_gain is not None else 0.0))
        return float(self._cfg_gain if self._cfg_gain is not None else 0.0)

    @gain.setter
    def gain(self, val) -> None:
        try:
            gv = float(val)
        except Exception:
            return
        self._cfg_gain = gv
        try:
            self._configs["gain"] = gv
        except Exception:
            pass
        if self._cam_open and not self._set_number("Gain", gv):
            self._log(logging.WARNING, "PyPylon:Gain not available")

    @property
    def gain_auto(self):
        if self._cam_open:
            return self._auto_mode_to_int("GainAuto", -1)
        if self._cfg_gain_auto is None:
            return -1
        return self._cfg_gain_auto

    @gain_auto.setter
    def gain_auto(self, val) -> None:
        self._cfg_gain_auto = val
        try:
            self._configs["gain_auto"] = val
        except Exception:
            pass
        if not self._cam_open:
            return
        # Accept bool/int or explicit symbolic
        if isinstance(val, str):
            if not self._enum_set("GainAuto", val):
                self._log(logging.WARNING, f"PyPylon:GainAuto mode {val} not available")
            return
        try:
            enabled = bool(int(val))
        except Exception:
            enabled = bool(val)
        if not self._apply_auto_mode("GainAuto", enabled):
            self._log(logging.WARNING, "PyPylon:GainAuto not available")

    @property
    def acquisition_mode(self) -> str:
        return self._enum_get("AcquisitionMode", str(self._cfg_acquisition_mode or ""))

    @acquisition_mode.setter
    def acquisition_mode(self, mode: str) -> None:
        self._cfg_acquisition_mode = str(mode)
        try:
            self._configs["acquisition_mode"] = str(mode)
        except Exception:
            pass
        if self._cam_open and (not self._set_enum_best_effort("AcquisitionMode", str(mode))):
            self._log(logging.WARNING, f"PyPylon:AcquisitionMode {mode} not available")

    @property
    def exposure_mode(self) -> str:
        return self._enum_get("ExposureMode", str(self._cfg_exposure_mode or ""))

    @exposure_mode.setter
    def exposure_mode(self, mode: str) -> None:
        self._cfg_exposure_mode = str(mode)
        try:
            self._configs["exposure_mode"] = str(mode)
        except Exception:
            pass
        if self._cam_open and (not self._set_enum_best_effort("ExposureMode", str(mode))):
            self._log(logging.WARNING, f"PyPylon:ExposureMode {mode} not available")

    @property
    def acquisition_frame_rate_enable(self) -> bool:
        return self._get_bool(
            "AcquisitionFrameRateEnable",
            bool(self._cfg_acquisition_frame_rate_enable) if self._cfg_acquisition_frame_rate_enable is not None else False,
        )

    @acquisition_frame_rate_enable.setter
    def acquisition_frame_rate_enable(self, enabled: bool) -> None:
        self._cfg_acquisition_frame_rate_enable = bool(enabled)
        try:
            self._configs["acquisition_frame_rate_enable"] = bool(enabled)
        except Exception:
            pass
        if self._cam_open and not self._set_bool("AcquisitionFrameRateEnable", bool(enabled)):
            self._log(logging.WARNING, "PyPylon:AcquisitionFrameRateEnable not available")

    @property
    def gamma(self) -> float:
        if self._cfg_gamma is None:
            self._cfg_gamma = 1.0
        if self._cam_open:
            self._cfg_gamma = float(self._get_number("Gamma", float(self._cfg_gamma)))
        return float(self._cfg_gamma)

    @gamma.setter
    def gamma(self, value: float) -> None:
        try:
            gv = float(value)
        except Exception:
            return
        self._cfg_gamma = gv
        try:
            self._configs["gamma"] = gv
        except Exception:
            pass
        if self._cam_open and not self._set_number("Gamma", gv):
            self._log(logging.WARNING, "PyPylon:Gamma not available")

    @property
    def gamma_enable(self) -> bool:
        if self._cfg_gamma_enable is None:
            self._cfg_gamma_enable = False
        if self._cam_open:
            self._cfg_gamma_enable = self._get_bool("GammaEnable", bool(self._cfg_gamma_enable))
        return bool(self._cfg_gamma_enable)

    @gamma_enable.setter
    def gamma_enable(self, enabled: bool) -> None:
        self._cfg_gamma_enable = bool(enabled)
        try:
            self._configs["gamma_enable"] = bool(enabled)
        except Exception:
            pass
        if self._cam_open and not self._set_bool("GammaEnable", bool(enabled)):
            self._log(logging.WARNING, "PyPylon:GammaEnable not available")

    @property
    def balance_white_auto(self):
        if self._cam_open:
            return self._auto_mode_to_int("BalanceWhiteAuto", -1)
        if self._cfg_balance_white_auto is None:
            return -1
        return self._cfg_balance_white_auto

    @balance_white_auto.setter
    def balance_white_auto(self, val) -> None:
        self._cfg_balance_white_auto = val
        try:
            self._configs["balance_white_auto"] = val
        except Exception:
            pass
        if not self._cam_open:
            return
        if isinstance(val, str):
            if not self._enum_set("BalanceWhiteAuto", val):
                self._log(logging.WARNING, f"PyPylon:BalanceWhiteAuto mode {val} not available")
            return
        try:
            enabled = bool(int(val))
        except Exception:
            enabled = bool(val)
        if not self._apply_auto_mode("BalanceWhiteAuto", enabled):
            self._log(logging.WARNING, "PyPylon:BalanceWhiteAuto not available")

    @property
    def binning(self):
        if self._cam_open:
            h = int(self._get_number("BinningHorizontal", 1, as_int=True))
            v = int(self._get_number("BinningVertical", 1, as_int=True))
            self._binning = (h, v)
        return self._binning

    @binning.setter
    def binning(self, val) -> None:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                h = int(val[0])
                v = int(val[1])
            except Exception:
                return
        else:
            try:
                h = v = int(val)
            except Exception:
                return
        self._binning = (h, v)
        try:
            self._configs["binning"] = (h, v)
        except Exception:
            pass
        if not self._cam_open or self.cam is None:
            return
        with self.cam_lock:
            was_grabbing = self.cam.IsGrabbing()
            if was_grabbing:
                self._stop_grabbing_locked()
            self._set_number("BinningHorizontal", h, as_int=True)
            self._set_number("BinningVertical", v, as_int=True)
            if was_grabbing:
                self._start_grabbing_locked()
            self._refresh_buffer_from_camera_locked()

    @property
    def offset(self):
        if self._cam_open:
            x = int(self._get_number("OffsetX", 0, as_int=True))
            y = int(self._get_number("OffsetY", 0, as_int=True))
            self._offset = (x, y)
        return self._offset

    @offset.setter
    def offset(self, val) -> None:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                x = int(val[0])
                y = int(val[1])
            except Exception:
                return
        else:
            try:
                x = y = int(val)
            except Exception:
                return
        self._offset = (x, y)
        try:
            self._configs["offset"] = (x, y)
        except Exception:
            pass
        if not self._cam_open or self.cam is None:
            return
        with self.cam_lock:
            was_grabbing = self.cam.IsGrabbing()
            if was_grabbing:
                self._stop_grabbing_locked()
            self._set_number("OffsetX", x, as_int=True)
            self._set_number("OffsetY", y, as_int=True)
            if was_grabbing:
                self._start_grabbing_locked()
            self._refresh_buffer_from_camera_locked()

    @property
    def adc(self) -> int:
        # Basler naming differs per model (`SensorBitDepth` etc.). Keep cached value.
        return int(self._adc)

    @adc.setter
    def adc(self, bits: int) -> None:
        try:
            b = int(bits)
        except Exception:
            return
        self._adc = b
        try:
            self._configs["adc"] = b
        except Exception:
            pass
        if not self._cam_open:
            return
        # Best effort for SensorBitDepth enums with values containing the bit count.
        options = self._enum_choices("SensorBitDepth")
        if options:
            target = None
            for opt in options:
                if str(b) in opt:
                    target = opt
                    break
            if target is not None:
                self._enum_set("SensorBitDepth", target)

    @property
    def pixel_format(self) -> str:
        if self._cam_open:
            pf = self._enum_get("PixelFormat", self._pixel_format_req or "")
            if pf:
                self._pixel_format_req = pf
        return str(self._pixel_format_req or "")

    @pixel_format.setter
    def pixel_format(self, fmt: str) -> None:
        s = str(fmt or "").strip()
        if not s:
            return
        self._pixel_format_req = s
        try:
            self._configs["pixel_format"] = s
        except Exception:
            pass
        if not self._cam_open or self.cam is None:
            return
        with self.cam_lock:
            was_grabbing = self.cam.IsGrabbing()
            if was_grabbing:
                self._stop_grabbing_locked()
            ok = self._enum_set("PixelFormat", s)
            if not ok:
                self._log(logging.WARNING, f"PyPylon:PixelFormat {s} not available")
            if was_grabbing:
                self._start_grabbing_locked()
            self._refresh_buffer_from_camera_locked()

    @property
    def trigin(self) -> int:
        return int(self._trigin)

    @trigin.setter
    def trigin(self, val) -> None:
        try:
            line = int(val)
        except Exception:
            return
        self._trigin = line
        self._cfg_trigin_present = True
        try:
            self._configs["trigin"] = line
        except Exception:
            pass
        if self._cam_open and self.cam is not None:
            with self.cam_lock:
                self._apply_trigger_input_locked()

    @property
    def trigout(self) -> int:
        return int(self._trigout)

    @trigout.setter
    def trigout(self, val) -> None:
        try:
            line = int(val)
        except Exception:
            return
        self._trigout = line
        self._cfg_trigout_present = True
        try:
            self._configs["trigout"] = line
        except Exception:
            pass
        if self._cam_open and self.cam is not None:
            with self.cam_lock:
                self._apply_trigger_output_locked()

    @property
    def ttlinv(self) -> bool:
        return bool(self._ttlinv)

    @ttlinv.setter
    def ttlinv(self, val) -> None:
        inv = bool(val)
        self._ttlinv = inv
        self._cfg_ttlinv_present = True
        try:
            self._configs["ttlinv"] = inv
        except Exception:
            pass
        if self._cam_open and self.cam is not None:
            with self.cam_lock:
                self._apply_trigger_input_locked()
                self._apply_trigger_output_locked()

    @property
    def convert_format(self) -> str:
        return str(self._convert_format)

    @convert_format.setter
    def convert_format(self, fmt: str) -> None:
        s = str(fmt or "native")
        if s not in self._CONVERT_FORMATS:
            self._log(logging.WARNING, f"PyPylon:Unsupported convert_format {s}; using native")
            s = "native"
        self._convert_format = s
        try:
            self._configs["convert_format"] = s
        except Exception:
            pass
        self._configure_converter()
        if self._cam_open and self.cam is not None:
            with self.cam_lock:
                self._refresh_buffer_from_camera_locked()

    # ------------------------------------------------------------------
    # Controls helpers
    # ------------------------------------------------------------------

    def get_control(self, name: str):
        key = str(name or "").strip()
        mapping = {
            "ExposureTime": self.exposure,
            "Exposure": self.exposure,
            "AutoExposure": self.autoexposure,
            "AcquisitionFrameRate": self.fps,
            "FPS": self.fps,
            "Width": self.width,
            "Height": self.height,
            "Resolution": self.resolution,
            "Gain": self.gain,
            "GainAuto": self.gain_auto,
            "BalanceWhiteAuto": self.balance_white_auto,
            "Gamma": self.gamma,
            "GammaEnable": self.gamma_enable,
            "Binning": self.binning,
            "Offset": self.offset,
            "PixelFormat": self.pixel_format,
            "ConvertFormat": self.convert_format,
            "Flip": self.flip,
            "TriggerIn": self.trigin,
            "TriggerOut": self.trigout,
            "TTLInverted": self.ttlinv,
            "StreamPolicy": self.stream_policy,
            "AcquisitionMode": self.acquisition_mode,
            "ExposureMode": self.exposure_mode,
            "AcquisitionFrameRateEnable": self.acquisition_frame_rate_enable,
        }
        if key in mapping:
            return mapping[key]
        if not self._cam_open or self.cam is None:
            return None
        p = self._param(key)
        if p is None:
            return None
        try:
            return p.GetValue()
        except Exception:
            try:
                return p.Value
            except Exception:
                return None

    def set_controls(self, controls: dict) -> bool:
        if not isinstance(controls, dict):
            return False
        ok_all = True
        for k, v in controls.items():
            try:
                if k in ("ExposureTime", "Exposure"):
                    self.exposure = float(v)
                elif k in ("AutoExposure",):
                    self.autoexposure = int(v)
                elif k in ("FPS", "AcquisitionFrameRate"):
                    self.fps = float(v)
                elif k in ("Gain",):
                    self.gain = float(v)
                elif k in ("GainAuto",):
                    self.gain_auto = v
                elif k in ("BalanceWhiteAuto",):
                    self.balance_white_auto = v
                elif k in ("Gamma",):
                    self.gamma = float(v)
                elif k in ("GammaEnable",):
                    self.gamma_enable = bool(v)
                elif k in ("Width",):
                    self.width = int(v)
                elif k in ("Height",):
                    self.height = int(v)
                elif k in ("Resolution", "Size"):
                    self.size = v
                elif k in ("Binning",):
                    self.binning = v
                elif k in ("Offset",):
                    self.offset = v
                elif k in ("ADC", "Adc", "adc"):
                    self.adc = int(v)
                elif k in ("PixelFormat",):
                    self.pixel_format = str(v)
                elif k in ("ConvertFormat",):
                    self.convert_format = str(v)
                elif k in ("Flip",):
                    self.flip = int(v)
                elif k in ("TriggerIn",):
                    self.trigin = int(v)
                elif k in ("TriggerOut",):
                    self.trigout = int(v)
                elif k in ("TTLInverted",):
                    self.ttlinv = bool(v)
                elif k in ("StreamPolicy",):
                    self.stream_policy = str(v)
                elif k in ("AcquisitionMode",):
                    self.acquisition_mode = str(v)
                elif k in ("ExposureMode",):
                    self.exposure_mode = str(v)
                elif k in ("AcquisitionFrameRateEnable",):
                    self.acquisition_frame_rate_enable = bool(v)
                elif k in ("ISPEnable",):
                    self.isp_enable = bool(v)
                else:
                    # Best-effort direct node set
                    if isinstance(v, bool):
                        ok = self._set_bool(k, v)
                    elif isinstance(v, (int, float)):
                        ok = self._set_number(k, v, as_int=isinstance(v, int))
                    else:
                        ok = self._enum_set(k, v)
                    if not ok:
                        ok_all = False
            except Exception:
                ok_all = False
        return ok_all

    # ------------------------------------------------------------------
    # Supported formats/options
    # ------------------------------------------------------------------

    def get_supported_main_color_formats(self) -> list[str]:
        formats = self._enum_choices("PixelFormat")
        if not formats:
            return []
        out = []
        for f in formats:
            fl = f.lower()
            if "bayer" in fl:
                continue
            out.append(f)
        return out

    def get_supported_raw_color_formats(self) -> list[str]:
        formats = self._enum_choices("PixelFormat")
        if not formats:
            return []
        out = []
        for f in formats:
            fl = f.lower()
            if ("bayer" in fl) or ("mono" in fl):
                out.append(f)
        return out

    def get_supported_raw_options(self) -> list[dict]:
        raw = self.get_supported_raw_color_formats()
        opts = []
        res = self.resolution
        fps = self.fps
        for fmt in raw:
            opts.append({"format": fmt, "resolution": res, "fps": fps})
        return opts

    def get_supported_main_options(self) -> list[dict]:
        main = self.get_supported_main_color_formats()
        opts = []
        res = self.resolution
        fps = self.fps
        for fmt in main:
            opts.append({"format": fmt, "resolution": res, "fps": fps})
        # CPU-side conversion targets
        opts.append({"convert_format": "native"})
        opts.append({"convert_format": "Mono8"})
        opts.append({"convert_format": "Mono16"})
        opts.append({"convert_format": "BGR8"})
        opts.append({"convert_format": "RGB8"})
        return opts

    def log_stream_options(self) -> None:
        try:
            self._log(logging.INFO, f"PyPylon:Model={self._metadata.get('model')} Serial={self._metadata.get('serial')}")
            self._log(logging.INFO, f"PyPylon:Resolution={self.resolution} FPS={self.fps:.3f}")
            self._log(logging.INFO, f"PyPylon:PixelFormat={self.pixel_format} ConvertFormat={self.convert_format}")
            mains = self.get_supported_main_color_formats()
            raws = self.get_supported_raw_color_formats()
            self._log(logging.INFO, f"PyPylon:Supported main formats: {mains}")
            self._log(logging.INFO, f"PyPylon:Supported raw formats: {raws}")
        except Exception:
            pass


__all__ = ["PylonCore"]

