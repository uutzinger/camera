###############################################################################
# FLIR Spinnaker camera core (PySpin / Spinnaker)
#
# Shared core used by both non-Qt threaded capture and Qt wrappers.
#
# Urs Utzinger
# GPT-5.2
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: SpinCore
#
# This is a non-threaded, non-Qt core wrapper intended to be composed by:
# - the threaded capture wrapper `spinCapture`
# - the Qt wrapper `spinCaptureQt`
#
# Public methods:
# - open_cam()         : Open and configure the camera, allocate frame buffer.
# - close_cam()        : Stop and close the camera, release frame buffer.
# - capture_array()    : tuple[np.ndarray | None, float | None]
#                        Capture a frame and timestamp from the configured stream.
# - trigger_software() : bool
#                        Execute a software trigger (when configured).
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
# - exposure, autoexposure, 
# - fps, 
# - flip, 
# - stream_policy
# - isp_enable, 
# - gain, gain_auto
# - acquisition_mode, exposure_mode, acquisition_frame_rate_enable
# - gamma, gamma_enable, balance_white_auto
# - binning, offset, adc, pixel_format
# - trigin, trigout, ttlinv
# - convert_format
#
# Processing pipeline (camera ISP vs host CPU):
# --------------------------------------------
# 1) Camera-side ISP (`isp_enable` -> `IspEnable`)
#    - Runs on the camera before transport to the host.
#    - Affects the pixel data delivered by `PixelFormat`.
# 2) Host-side conversion (`convert_format` + `color_processing`)
#    - Runs on the CPU via `PySpin.ImageProcessor.Convert(...)`.
#    - Used when `convert_format != 'native'`.
# 3) Host-side geometry (`output_res`, `flip`)
#    - Always runs on the CPU via OpenCV (`cv2.resize/rotate/flip`).
#
# Important interactions:
# - Disabling ISP does NOT disable CPU conversion/resize/flip.
# - `flip` and `output_res` always use OpenCV on the CPU.
# - `convert_format` runs on the CPU via `PySpin.ImageProcessor.Convert(...)`.
# - If you want display-ready color, set `convert_format='BGR8'` (or 'RGB8'),
#   especially when `pixel_format` is a Bayer format and ISP is disabled.
#
# ISP on/off summary:
# - ISP on: camera may output already-processed pixels; CPU steps still apply. Not ideal for max fps.
# - ISP off: camera output is more "raw"; use `convert_format` for debayer/color.
#
# Supported config parameters (configs dict):
# -------------------------------------------
#
# Stream policy applies a baseline first, then explicit overrides from config
# and setters are re-applied.
#
# Core capture / output:
# - camera_res: tuple[int,int]      Capture size (w,h)
# - output_res: tuple[int,int]      Optional CPU resize target, uses opencv
# - fps: float|int                  Requested framerate
# - flip: int                       0..7 (same enum as cv2Capture), uses opencv
# - stream_policy: str              'default'|'max_fps'|'auto'
# - buffersize: int                 FrameBuffer capacity (default: 32)
# - buffer_overwrite: bool          If True, FrameBuffer overwrites oldest frames when full
# - capture_timeout_ms: int         GetNextImage timeout in ms (default: 100)
#
# Sensor / format controls:
# - exposure: int|float             Exposure time in microseconds
# - autoexposure: int               -1 leave unchanged, 0 off, 1 on
# - isp_enable: bool                Enable/disable camera ISP when available
# - gain: float                     Manual gain
# - gain_auto: int|bool             -1 leave unchanged, 0 off, 1 on
# - binning: tuple[int,int]|int     Sensor binning (h,v)
# - offset: tuple[int,int]|int      ROI offset (x,y)
# - adc: int                        ADC bit depth (8,10,12,14)
# - pixel_format: str               Pixel format override (e.g. 'Mono8', 'Mono16', 'BayerRG8')
# - convert_format: str             Conversion target ('native', 'Mono8', 'Mono16', 'BGR8', 'RGB8'), uses SpinnaKer ImageProcessor
# - color_processing: str           ImageProcessor color algorithm (default: 'HQ_LINEAR')
# - acquisition_mode: str           Acquisition mode override (e.g. 'continuous')
# - exposure_mode: str              Exposure mode override (e.g. 'timed')
# - acquisition_frame_rate_enable: bool
# - gamma: float
# - gamma_enable: bool
# - balance_white_auto: int|bool    Auto white balance when available
#
# Trigger / IO:
# - trigin: int                     -1 software trigger, else line index (0..3)
# - trigout: int                    -1 disable, else line index (1..3 typical)
# - ttlinv: bool                    Invert trigger polarity
###############################################################################

from __future__ import annotations

from threading import Lock
from queue import Queue  # logging
import logging
import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .framebuffer import FrameBuffer

try:
    import PySpin
except Exception:  # pragma: no cover
    PySpin = None  # type: ignore


class SpinCore:
    """Core PySpin wrapper."""

    _ADC_ENUM = {
        8: "AdcBitDepth_Bit8",
        10: "AdcBitDepth_Bit10",
        12: "AdcBitDepth_Bit12",
        14: "AdcBitDepth_Bit14",
    }

    _DEFAULT_COLOR_ALGO = "SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR"

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

        # Config normalization
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

        # Track which controls were explicitly requested in configs/args so we
        # can let stream policy provide defaults without overriding user intent.
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
        self._color_processing = str(self._configs.get("color_processing", "HQ_LINEAR") or "HQ_LINEAR")

        self._trigout = int(self._configs.get("trigout", -1))
        self._ttlinv = bool(self._configs.get("ttlinv", False))
        self._trigin = int(self._configs.get("trigin", -1))
        self._cfg_trigout_present = "trigout" in self._configs
        self._cfg_ttlinv_present = "ttlinv" in self._configs
        self._cfg_trigin_present = "trigin" in self._configs

        # Optional controls that were previously hard-coded for performance.
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

        # Frame buffer management
        self._buffer_capacity = int(self._configs.get("buffersize", 32))
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        self._buffer: FrameBuffer | None = None

        # Processing flags
        self._needs_cpu_resize = (self._output_width > 0 and self._output_height > 0)
        self._needs_cpu_flip = (self._flip_method != 0)

        # Conversion
        self._processor = None
        self._convert_pixel_format = None
        self._is_color_output = False
        self._buffer_dtype = np.uint8
        self._buffer_channels = 1

        # Runtime
        self.system = None
        self.cam_list = None
        self.cam = None
        self.cam_open = False
        self.cam_lock = Lock()
        self._metadata: dict | None = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str) -> None:
        if self.log is None:
            return
        try:
            if not self.log.full():
                self.log.put_nowait((level, message))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_stream_policy(value) -> str:
        try:
            _tmp = str(value or "default").strip().lower()
        except Exception:
            _tmp = "default"
        if _tmp in ("max_fps", "maximize_fps", "maximize", "performance", "fast"):
            return "max_fps"
        if _tmp in ("auto", "automatic", "quality"):
            return "auto"
        return "default"

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_writable(node) -> bool:
        try:
            return node is not None and node.GetAccessMode() == PySpin.RW  # type: ignore[name-defined]
        except Exception:
            return False

    @staticmethod
    def _getattr_enum(name: str):
        if PySpin is None:
            return None
        return getattr(PySpin, name, None)

    def _pixel_format_const(self, fmt: str):
        if PySpin is None:
            return None
        name = f"PixelFormat_{fmt}"
        return getattr(PySpin, name, None)

    def _pixel_format_to_str(self, value) -> str:
        if PySpin is None:
            return "unknown"
        # Minimal list of common formats we expect to encounter.
        candidates = (
            "Mono8",
            "Mono10",
            "Mono10p",
            "Mono12",
            "Mono12p",
            "Mono16",
            "BayerRG8",
            "BayerRG10",
            "BayerRG12",
            "BayerRG16",
            "BayerBG8",
            "BayerBG10",
            "BayerBG12",
            "BayerBG16",
            "BGR8",
            "RGB8",
        )
        for cand in candidates:
            const = self._pixel_format_const(cand)
            if const is not None and const == value:
                return cand
        return "unknown"

    @staticmethod
    def _color_processing_const(name: str):
        if PySpin is None:
            return None
        _tmp = str(name or "").strip().upper()
        if not _tmp:
            return getattr(PySpin, SpinCore._DEFAULT_COLOR_ALGO, None)
        if not _tmp.startswith("SPINNAKER_COLOR_PROCESSING_ALGORITHM_"):
            _tmp = f"SPINNAKER_COLOR_PROCESSING_ALGORITHM_{_tmp}"
        return getattr(PySpin, _tmp, getattr(PySpin, SpinCore._DEFAULT_COLOR_ALGO, None))

    # ------------------------------------------------------------------
    # Stream policy
    # ------------------------------------------------------------------

    def _set_isp_enable(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_isp_enable = bool(value)
            try:
                self._configs["isp_enable"] = bool(value)
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "IspEnable", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:IspEnable not supported")
            return
        if self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(bool(value))
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:IspEnable set failed: {exc}")

    def _set_gain_auto(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_gain_auto = value
            try:
                self._configs["gain_auto"] = value
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "GainAuto", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:GainAuto not supported")
            return
        self._set_gain_selector_all()
        ga_cont = self._getattr_enum("GainAuto_Continuous")
        ga_once = self._getattr_enum("GainAuto_Once")
        ga_off  = self._getattr_enum("GainAuto_Off")
        target  = ga_cont if bool(value) else ga_off
        if isinstance(value, str):
            if value.strip().lower() == "once" and ga_once is not None:
                target = ga_once
        if target is not None and self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(target)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:GainAuto set failed: {exc}")

    def _set_gain(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_gain = float(value)
            try:
                self._configs["gain"] = float(value)
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "Gain", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:Gain not supported")
            return
        # Manual gain implies gain auto off when available.
        self._set_gain_auto(0, explicit=False)
        self._set_gain_selector_all()
        if self._is_writable(node):
            try:
                v = max(node.GetMin(), min(node.GetMax(), float(value)))
                with self.cam_lock:
                    node.SetValue(v)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:Gain set failed: {exc}")

    def _set_autoexposure(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_autoexposure_present = True
            self._autoexposure = int(value)
            try:
                self._configs["autoexposure"] = int(value)
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "ExposureAuto", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:ExposureAuto not supported")
            return
        ae_cont = self._getattr_enum("ExposureAuto_Continuous")
        ae_once = self._getattr_enum("ExposureAuto_Once")
        ae_off = self._getattr_enum("ExposureAuto_Off")
        target = ae_cont if bool(value) else ae_off
        if isinstance(value, str):
            if value.strip().lower() == "once" and ae_once is not None:
                target = ae_once
        if target is not None and self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(target)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:ExposureAuto set failed: {exc}")

    def _set_acquisition_mode(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_acquisition_mode = value
            try:
                self._configs["acquisition_mode"] = value
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "AcquisitionMode", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:AcquisitionMode not supported")
            return
        mode = str(value or "").strip().lower()
        enum_name = {
            "continuous": "AcquisitionMode_Continuous",
            "single": "AcquisitionMode_SingleFrame",
            "singleframe": "AcquisitionMode_SingleFrame",
            "multi": "AcquisitionMode_MultiFrame",
            "multiframe": "AcquisitionMode_MultiFrame",
        }.get(mode, None)
        enum_val = self._getattr_enum(enum_name) if enum_name else None
        if enum_val is None:
            if explicit:
                self._log(logging.WARNING, f"PySpin:Unknown acquisition_mode '{value}'")
            return
        if self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(enum_val)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:AcquisitionMode set failed: {exc}")

    def _set_exposure_mode(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_exposure_mode = value
            try:
                self._configs["exposure_mode"] = value
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "ExposureMode", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:ExposureMode not supported")
            return
        mode = str(value or "").strip().lower()
        enum_name = {
            "timed": "ExposureMode_Timed",
            "triggerwidth": "ExposureMode_TriggerWidth",
            "trigger_width": "ExposureMode_TriggerWidth",
        }.get(mode, None)
        enum_val = self._getattr_enum(enum_name) if enum_name else None
        if enum_val is None:
            if explicit:
                self._log(logging.WARNING, f"PySpin:Unknown exposure_mode '{value}'")
            return
        if self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(enum_val)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:ExposureMode set failed: {exc}")

    def _set_acquisition_frame_rate_enable(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_acquisition_frame_rate_enable = bool(value)
            try:
                self._configs["acquisition_frame_rate_enable"] = bool(value)
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "AcquisitionFrameRateEnable", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:AcquisitionFrameRateEnable not supported")
            return
        if self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(bool(value))
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:AcquisitionFrameRateEnable set failed: {exc}")

    def _set_gamma_enable(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_gamma_enable = bool(value)
            try:
                self._configs["gamma_enable"] = bool(value)
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "GammaEnable", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:GammaEnable not supported")
            return
        if self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(bool(value))
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:GammaEnable set failed: {exc}")

    def _set_gamma(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_gamma = float(value)
            try:
                self._configs["gamma"] = float(value)
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "Gamma", None)
        if node is None:
            if explicit:
                self._log(logging.WARNING, "PySpin:Gamma not supported")
            return
        # Setting gamma implies gamma enable on when available.
        self._set_gamma_enable(True, explicit=False)
        if self._is_writable(node):
            try:
                v = max(node.GetMin(), min(node.GetMax(), float(value)))
                with self.cam_lock:
                    node.SetValue(v)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:Gamma set failed: {exc}")

    def _set_balance_white_auto(self, value, *, explicit: bool) -> None:
        if explicit:
            self._cfg_balance_white_auto = value
            try:
                self._configs["balance_white_auto"] = value
            except Exception:
                pass
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "BalanceWhiteAuto", None)
        if node is None:
            return
        bw_cont = self._getattr_enum("BalanceWhiteAuto_Continuous")
        bw_once = self._getattr_enum("BalanceWhiteAuto_Once")
        bw_off = self._getattr_enum("BalanceWhiteAuto_Off")
        target = bw_cont if bool(value) else bw_off
        if isinstance(value, str):
            if value.strip().lower() == "once" and bw_once is not None:
                target = bw_once
        if target is not None and self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(target)
            except Exception as exc:
                if explicit:
                    self._log(logging.WARNING, f"PySpin:BalanceWhiteAuto set failed: {exc}")

    def _apply_stream_policy(self) -> None:
        """Apply stream policy defaults before user overrides."""
        if not self.cam_open:
            return
        policy = self._stream_policy
        self._log(logging.INFO, f"PySpin:StreamPolicy:{policy}")

        if policy == "max_fps":
            # Performance-first defaults; users can override via configs/setters.
            self._set_isp_enable(False, explicit=False)
            self._set_gain_auto(0, explicit=False)
            self._set_gain(1.0, explicit=False)
            self._set_autoexposure(0, explicit=False)
            self._set_exposure_mode("timed", explicit=False)
            self._set_acquisition_mode("continuous", explicit=False)
            self._set_acquisition_frame_rate_enable(True, explicit=False)
            self._set_gamma(1.0, explicit=False)
            self._set_gamma_enable(False, explicit=False)
            self._set_balance_white_auto(0, explicit=False)
        elif policy == "auto":
            # Quality/automation defaults; still best-effort.
            self._set_isp_enable(True, explicit=False)
            self._set_autoexposure(1, explicit=False)
            self._set_gain_auto(1, explicit=False)
            self._set_balance_white_auto(1, explicit=False)
            self._set_acquisition_mode("continuous", explicit=False)
            self._set_gamma_enable(True, explicit=False)

    def _apply_config_overrides(self) -> None:
        """Apply explicit user-requested overrides after stream policy."""
        if not self.cam_open:
            return

        # Optional performance/quality controls
        if self._cfg_isp_enable is not None:
            self.isp_enable = bool(self._cfg_isp_enable)
        if self._cfg_acquisition_mode is not None:
            self.acquisition_mode = self._cfg_acquisition_mode
        if self._cfg_exposure_mode is not None:
            self.exposure_mode = self._cfg_exposure_mode
        if self._cfg_acquisition_frame_rate_enable is not None:
            self.acquisition_frame_rate_enable = bool(self._cfg_acquisition_frame_rate_enable)
        if self._cfg_gamma_enable is not None:
            self.gamma_enable = bool(self._cfg_gamma_enable)
        if self._cfg_gamma is not None:
            self.gamma = self._cfg_gamma
        if self._cfg_gain_auto is not None:
            self.gain_auto = self._cfg_gain_auto
        if self._cfg_gain is not None:
            self.gain = self._cfg_gain
        if self._cfg_balance_white_auto is not None:
            self.balance_white_auto = self._cfg_balance_white_auto

        # Legacy controls (apply only when explicitly requested)
        if self._cfg_autoexposure_present:
            self.autoexposure = self._autoexposure
        if self._cfg_exposure_present and (self._exposure is not None) and (self._exposure > 0):
            self.exposure = self._exposure
        if self._cfg_fps_present:
            self.fps = self._framerate

        # Trigger/IO overrides
        if self._cfg_ttlinv_present:
            self.ttlinv = self._ttlinv
        if self._cfg_trigin_present:
            self.trigin = self._trigin
        if self._cfg_trigout_present:
            self.trigout = self._trigout

        # Pixel format override should happen after ADC selection.
        if self._pixel_format_req:
            self.pixel_format = self._pixel_format_req

    # ------------------------------------------------------------------
    # Conversion / buffer sizing
    # ------------------------------------------------------------------

    def _configure_conversion(self) -> None:
        self._processor = None
        self._convert_pixel_format = None

        target = str(self._convert_format or "native").strip()
        if not target or target.lower() == "native":
            self._is_color_output = False
            return

        fmt_const = self._pixel_format_const(target)
        if fmt_const is None:
            self._log(logging.WARNING, f"PySpin:Unknown convert_format '{target}', using native")
            return

        try:
            processor = PySpin.ImageProcessor()  # type: ignore[union-attr]
            algo = self._color_processing_const(self._color_processing)
            if algo is not None:
                processor.SetColorProcessing(algo)
            self._processor = processor
            self._convert_pixel_format = fmt_const
        except Exception as exc:
            self._processor = None
            self._convert_pixel_format = None
            self._log(logging.WARNING, f"PySpin:ImageProcessor init failed: {exc}")
            return

        target_u = target.upper()
        self._is_color_output = target_u in ("BGR8", "RGB8")

    def _expected_buffer_spec(self) -> tuple[tuple[int, int], int, np.dtype]:
        w, h = self._effective_output_size()
        fmt = str(self._convert_format or "native").strip()
        if not fmt or fmt.lower() == "native":
            # Native output: infer from the current pixel format when possible.
            pf = self.pixel_format if self.cam_open else ""
            pf_u = str(pf or "").upper()
            if pf_u in ("BGR8", "RGB8"):
                return (w, h), 3, np.uint8
            if ("16" in pf_u) or (self._adc > 8):
                return (w, h), 1, np.uint16
            return (w, h), 1, np.uint8

        fmt_u = fmt.upper()
        if fmt_u in ("BGR8", "RGB8"):
            return (w, h), 3, np.uint8
        if fmt_u == "MONO16":
            return (w, h), 1, np.uint16
        # Default to Mono8 for any other explicit format.
        return (w, h), 1, np.uint8

    def _effective_output_size(self) -> tuple[int, int]:
        if self._output_width > 0 and self._output_height > 0:
            w, h = self._output_width, self._output_height
        else:
            w, h = self._capture_width, self._capture_height
        if self._flip_method in (1, 3, 5, 7):
            return (h, w)
        return (w, h)

    def _allocate_buffer(self) -> None:
        (w, h), channels, dtype = self._expected_buffer_spec()
        self._buffer_dtype = dtype
        self._buffer_channels = channels
        if channels > 1:
            frame_shape = (h, w, channels)
        else:
            frame_shape = (h, w)
        self._buffer = FrameBuffer(
            capacity=self._buffer_capacity,
            frame_shape=frame_shape,
            dtype=dtype,
            overwrite=self._buffer_overwrite,
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_array(self) -> tuple[np.ndarray | None, float | None]:
        if not self.cam_open or self.cam is None:
            return None, None

        image_result = None
        try:
            with self.cam_lock:
                image_result = self.cam.GetNextImage(self._capture_timeout_ms)

            if image_result is None:
                return None, None

            if image_result.IsIncomplete():
                self._log(
                    logging.WARNING,
                    f"PySpin:Incomplete image: {image_result.GetImageStatus()}",
                )
                return None, None

            # Timestamp from camera is in nanoseconds.
            try:
                ts_ms = float(image_result.GetTimeStamp()) / 1_000_000.0
            except Exception:
                ts_ms = time.perf_counter() * 1000.0

            img = None
            if self._processor is not None and self._convert_pixel_format is not None:
                try:
                    converted = self._processor.Convert(image_result, self._convert_pixel_format)
                    img = converted.GetNDArray()
                except Exception as exc:
                    self._log(logging.WARNING, f"PySpin:Convert failed: {exc}")
                    img = image_result.GetNDArray()
            else:
                img = image_result.GetNDArray()

            if img is None:
                return None, None

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
            self._log(logging.ERROR, f"PySpin:GetNextImage failed: {exc}")
            return None, None
        finally:
            if image_result is not None:
                try:
                    image_result.Release()
                except Exception:
                    self._log(logging.WARNING, "PySpin:Could not release image")

    def trigger_software(self) -> bool:
        """Execute a software trigger when supported."""
        if not self.cam_open or self.cam is None:
            return False
        try:
            with self.cam_lock:
                if self._is_writable(self.cam.TriggerSource):
                    software = self._getattr_enum("TriggerSource_Software")
                    if software is not None:
                        self.cam.TriggerSource.SetValue(software)
                if self._is_writable(self.cam.TriggerSoftware):
                    self.cam.TriggerSoftware()
                else:
                    self.cam.TriggerSoftware()
            return True
        except Exception as exc:
            self._log(logging.WARNING, f"PySpin:TriggerSoftware failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open_cam(self) -> bool:
        if PySpin is None:
            self._log(logging.CRITICAL, "PySpin is not installed")
            self.cam_open = False
            return False

        # Always close first to ensure a clean state.
        self.close_cam()

        try:
            self.system = PySpin.System.GetInstance()
            version = self.system.GetLibraryVersion()
            self._log(
                logging.INFO,
                f"PySpin:Driver:Version:{version.major}.{version.minor}.{version.type}.{version.build}",
            )

            self.cam_list = self.system.GetCameras()
            num_cameras = self.cam_list.GetSize()
            self._log(logging.INFO, f"PySpin:Number of Cameras:{num_cameras}")
            if num_cameras == 0:
                self._log(logging.CRITICAL, "PySpin:No Cameras Found")
                self.close_cam()
                return False

            self.cam = self.cam_list[self._camera_num]

            # Log device information (best-effort).
            try:
                nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
                node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode("DeviceInformation"))
                if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                    for feature in node_device_information.GetFeatures():
                        node_feature = PySpin.CValuePtr(feature)
                        if PySpin.IsReadable(node_feature):
                            self._log(
                                logging.INFO,
                                f"PySpin:CameraFeature:{node_feature.GetName()}={node_feature.ToString()}",
                            )
            except Exception:
                pass

            self.cam.Init()
            self.cam_open = True

            # Sensor setup order matters: ADC/binning -> size -> offset.
            try:
                if self._is_writable(self.cam.BinningSelector):
                    selector = self._getattr_enum("BinningSelector_Sensor")
                    if selector is not None:
                        self.cam.BinningSelector.SetValue(selector)
                if self._is_writable(self.cam.BinningVerticalMode):
                    mode_sum = self._getattr_enum("BinningVerticalMode_Sum")
                    if mode_sum is not None:
                        self.cam.BinningVerticalMode.SetValue(mode_sum)
                if self._is_writable(self.cam.BinningHorizontalMode):
                    mode_sum = self._getattr_enum("BinningHorizontalMode_Sum")
                    if mode_sum is not None:
                        self.cam.BinningHorizontalMode.SetValue(mode_sum)
            except Exception:
                pass

            # Apply baseline sensor geometry first.
            self.adc = self._adc
            self.binning = self._binning
            self.resolution = self._camera_res
            self.offset = self._offset

            # Apply stream policy defaults, then explicit user overrides.
            self._apply_stream_policy()
            self._apply_config_overrides()

            # Configure conversion and allocate buffer before acquisition.
            self._configure_conversion()
            self._allocate_buffer()

            with self.cam_lock:
                self.cam.BeginAcquisition()

            # Kick software trigger once when explicitly configured.
            if self._cfg_trigin_present and self._trigin == -1:
                self.trigger_software()

            pf_str = self.pixel_format if self.cam_open else "unknown"
            w, h = self._effective_output_size()
            adc_str = str(self._adc) if self._adc and self._adc > 0 else "default"
            self._log(
                logging.INFO,
                f"PySpin:Open summary size={self._capture_width}x{self._capture_height} "
                f"pixel_format={pf_str} adc={adc_str} policy={self._stream_policy} "
                f"convert={self._convert_format} output={w}x{h}",
            )
            return True

        except Exception as exc:
            self._log(logging.CRITICAL, f"PySpin:Failed to open camera: {exc}")
            self.close_cam()
            return False

    def close_cam(self) -> None:
        """Stop acquisition and release PySpin resources (idempotent)."""
        try:
            cam = self.cam
            if cam is not None:
                try:
                    with self.cam_lock:
                        cam.EndAcquisition()
                except Exception:
                    pass
                try:
                    cam.DeInit()
                except Exception:
                    pass
        finally:
            self.cam = None
            self.cam_open = False
            self._buffer = None
            self._metadata = None

            cam_list = self.cam_list
            if cam_list is not None:
                try:
                    cam_list.Clear()
                except Exception:
                    pass
            self.cam_list = None

            system = self.system
            if system is not None:
                try:
                    system.ReleaseInstance()
                except Exception:
                    pass
            self.system = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> FrameBuffer | None:
        return self._buffer

    @property
    def metadata(self) -> dict:
        return self._metadata or {}

    def _set_gain_selector_all(self) -> None:
        if not self.cam_open or self.cam is None:
            return
        node = getattr(self.cam, "GainSelector", None)
        selector_all = self._getattr_enum("GainSelector_All")
        if node is not None and selector_all is not None and self._is_writable(node):
            try:
                with self.cam_lock:
                    node.SetValue(selector_all)
            except Exception:
                pass

    @property
    def width(self) -> int:
        if self.cam_open and self.cam is not None:
            return int(self.cam.Width.GetValue())
        return -1
    @width.setter
    def width(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"PySpin:Width not changed:{val}")
            return
        if self.cam_open and self.cam is not None and self._is_writable(self.cam.Width):
            val = min(max(val, self.cam.Width.GetMin()), self.cam.Width.GetMax())
            with self.cam_lock:
                self.cam.Width.SetValue(int(val))
            self._capture_width = int(self.cam.Width.GetValue())
        else:
            self._log(logging.ERROR, "PySpin:Failed to set width")

    @property
    def height(self) -> int:
        if self.cam_open and self.cam is not None:
            return int(self.cam.Height.GetValue())
        return -1
    @height.setter
    def height(self, val) -> None:
        if (val is None) or (val == -1):
            self._log(logging.WARNING, f"PySpin:Height not changed:{val}")
            return
        if self.cam_open and self.cam is not None and self._is_writable(self.cam.Height):
            val = min(max(val, self.cam.Height.GetMin()), self.cam.Height.GetMax())
            with self.cam_lock:
                self.cam.Height.SetValue(int(val))
            self._capture_height = int(self.cam.Height.GetValue())
        else:
            self._log(logging.ERROR, "PySpin:Failed to set height")

    @property
    def resolution(self) -> tuple[int, int]:
        if self.cam_open and self.cam is not None:
            return (int(self.cam.Width.GetValue()), int(self.cam.Height.GetValue()))
        return (-1, -1)
    @resolution.setter
    def resolution(self, val) -> None:
        if val is None or not self.cam_open or self.cam is None:
            return
        if isinstance(val, (tuple, list)) and len(val) > 1:
            w = min(max(val[0], self.cam.Width.GetMin()), self.cam.Width.GetMax())
            h = min(max(val[1], self.cam.Height.GetMin()), self.cam.Height.GetMax())
        else:
            w = min(max(val, self.cam.Width.GetMin()), self.cam.Width.GetMax())
            h = min(max(val, self.cam.Height.GetMin()), self.cam.Height.GetMax())
        if self._is_writable(self.cam.Width):
            with self.cam_lock:
                self.cam.Width.SetValue(int(w))
        if self._is_writable(self.cam.Height):
            with self.cam_lock:
                self.cam.Height.SetValue(int(h))
        self._capture_width = int(self.cam.Width.GetValue())
        self._capture_height = int(self.cam.Height.GetValue())

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
        except Exception:
            self._log(logging.ERROR, "Invalid size values")
            return
        self._camera_res = (width, height)
        self._capture_width, self._capture_height = self._camera_res
        if self.cam_open:
            self.close_cam()
            self.open_cam()

    @property
    def flip(self) -> int:
        try:
            return int(self._flip_method)
        except Exception:
            return 0
    @flip.setter
    def flip(self, value) -> None:
        try:
            f = int(value)
        except Exception:
            self._log(logging.ERROR, "Invalid flip value")
            return
        if f < 0 or f > 7:
            self._log(logging.ERROR, "flip must be in range 0..7")
            return
        self._flip_method = f
        self._needs_cpu_flip = (self._flip_method != 0)

    @property
    def stream_policy(self) -> str:
        return str(self._stream_policy)

    @stream_policy.setter
    def stream_policy(self, value) -> None:
        policy = self._normalize_stream_policy(value)
        if policy == self._stream_policy:
            return
        self._stream_policy = policy
        try:
            self._configs["stream_policy"] = policy
        except Exception:
            pass
        if self.cam_open:
            self.close_cam()
            self.open_cam()

    @property
    def isp_enable(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "IspEnable", None)
            if node is not None:
                try:
                    return bool(node.GetValue())
                except Exception:
                    pass
        if self._cfg_isp_enable is not None:
            return bool(self._cfg_isp_enable)
        return -1

    @isp_enable.setter
    def isp_enable(self, value) -> None:
        if value is None:
            return
        self._set_isp_enable(value, explicit=True)

    @property
    def gain(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "Gain", None)
            if node is not None:
                try:
                    return float(node.GetValue())
                except Exception:
                    pass
        if self._cfg_gain is not None:
            return float(self._cfg_gain)
        return float("nan")

    @gain.setter
    def gain(self, value) -> None:
        if value is None or value == -1:
            return
        self._set_gain(value, explicit=True)

    @property
    def gain_auto(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "GainAuto", None)
            if node is not None:
                try:
                    ga_val = node.GetValue()
                    ga_cont = self._getattr_enum("GainAuto_Continuous")
                    ga_once = self._getattr_enum("GainAuto_Once")
                    if ga_val in (ga_cont, ga_once):
                        return 1
                    return 0
                except Exception:
                    pass
        if self._cfg_gain_auto is not None:
            try:
                return 1 if bool(self._cfg_gain_auto) else 0
            except Exception:
                return self._cfg_gain_auto
        return -1

    @gain_auto.setter
    def gain_auto(self, value) -> None:
        if value is None or value == -1:
            return
        self._set_gain_auto(value, explicit=True)

    @property
    def acquisition_mode(self) -> str:
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "AcquisitionMode", None)
            if node is not None:
                try:
                    val = node.GetValue()
                    if val == self._getattr_enum("AcquisitionMode_Continuous"):
                        return "continuous"
                    if val == self._getattr_enum("AcquisitionMode_SingleFrame"):
                        return "singleframe"
                    if val == self._getattr_enum("AcquisitionMode_MultiFrame"):
                        return "multiframe"
                except Exception:
                    pass
        if self._cfg_acquisition_mode is not None:
            return str(self._cfg_acquisition_mode)
        return "unknown"

    @acquisition_mode.setter
    def acquisition_mode(self, value) -> None:
        if value is None:
            return
        self._set_acquisition_mode(value, explicit=True)

    @property
    def exposure_mode(self) -> str:
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "ExposureMode", None)
            if node is not None:
                try:
                    val = node.GetValue()
                    if val == self._getattr_enum("ExposureMode_Timed"):
                        return "timed"
                    if val == self._getattr_enum("ExposureMode_TriggerWidth"):
                        return "triggerwidth"
                except Exception:
                    pass
        if self._cfg_exposure_mode is not None:
            return str(self._cfg_exposure_mode)
        return "unknown"

    @exposure_mode.setter
    def exposure_mode(self, value) -> None:
        if value is None:
            return
        self._set_exposure_mode(value, explicit=True)

    @property
    def acquisition_frame_rate_enable(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "AcquisitionFrameRateEnable", None)
            if node is not None:
                try:
                    return bool(node.GetValue())
                except Exception:
                    pass
        if self._cfg_acquisition_frame_rate_enable is not None:
            return bool(self._cfg_acquisition_frame_rate_enable)
        return -1

    @acquisition_frame_rate_enable.setter
    def acquisition_frame_rate_enable(self, value) -> None:
        if value is None:
            return
        self._set_acquisition_frame_rate_enable(value, explicit=True)

    @property
    def gamma_enable(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "GammaEnable", None)
            if node is not None:
                try:
                    return bool(node.GetValue())
                except Exception:
                    pass
        if self._cfg_gamma_enable is not None:
            return bool(self._cfg_gamma_enable)
        return -1

    @gamma_enable.setter
    def gamma_enable(self, value) -> None:
        if value is None:
            return
        self._set_gamma_enable(value, explicit=True)

    @property
    def gamma(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "Gamma", None)
            if node is not None:
                try:
                    return float(node.GetValue())
                except Exception:
                    pass
        if self._cfg_gamma is not None:
            return float(self._cfg_gamma)
        return float("nan")

    @gamma.setter
    def gamma(self, value) -> None:
        if value is None or value == -1:
            return
        self._set_gamma(value, explicit=True)

    @property
    def balance_white_auto(self):
        if self.cam_open and self.cam is not None:
            node = getattr(self.cam, "BalanceWhiteAuto", None)
            if node is not None:
                try:
                    val = node.GetValue()
                    bw_cont = self._getattr_enum("BalanceWhiteAuto_Continuous")
                    bw_once = self._getattr_enum("BalanceWhiteAuto_Once")
                    if val in (bw_cont, bw_once):
                        return 1
                    return 0
                except Exception:
                    pass
        if self._cfg_balance_white_auto is not None:
            try:
                return 1 if bool(self._cfg_balance_white_auto) else 0
            except Exception:
                return self._cfg_balance_white_auto
        return -1

    @balance_white_auto.setter
    def balance_white_auto(self, value) -> None:
        if value is None or value == -1:
            return
        self._set_balance_white_auto(value, explicit=True)

    @property
    def exposure(self):
        if self.cam_open and self.cam is not None:
            return self.cam.ExposureTime.GetValue()
        return float("nan")
    @exposure.setter
    def exposure(self, val) -> None:
        if (val is None) or (val == -1):
            return
        try:
            self._exposure = float(val)
        except Exception:
            return
        self._cfg_exposure_present = True
        try:
            self._configs["exposure"] = self._exposure
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        # Manual exposure requires timed mode and auto exposure off.
        self._set_exposure_mode("timed", explicit=False)
        self._set_autoexposure(0, explicit=False)
        if self._is_writable(self.cam.ExposureTime):
            with self.cam_lock:
                self.cam.ExposureTime.SetValue(
                    max(self.cam.ExposureTime.GetMin(), min(self.cam.ExposureTime.GetMax(), self._exposure))
                )
            self._exposure = float(self.cam.ExposureTime.GetValue())

    @property
    def autoexposure(self):
        if not self.cam_open or self.cam is None:
            return -1
        try:
            ae_val = self.cam.ExposureAuto.GetValue()
            ae_cont = self._getattr_enum("ExposureAuto_Continuous")
            ae_once = self._getattr_enum("ExposureAuto_Once")
            if ae_val in (ae_cont, ae_once):
                return 1
        except Exception:
            pass
        return 0
    @autoexposure.setter
    def autoexposure(self, val) -> None:
        if (val is None) or (val == -1):
            return
        self._set_autoexposure(val, explicit=True)

    @property
    def fps(self):
        if self.cam_open and self.cam is not None:
            return self.cam.AcquisitionFrameRate.GetValue()
        return float("nan")
    @fps.setter
    def fps(self, val) -> None:
        if (val is None) or (val == -1):
            return
        try:
            self._framerate = float(val)
        except Exception:
            return
        self._cfg_fps_present = True
        try:
            self._configs["fps"] = self._framerate
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        # Frame-rate control typically requires the enable gate to be on.
        self._set_acquisition_frame_rate_enable(True, explicit=False)
        if self._is_writable(self.cam.AcquisitionFrameRate):
            with self.cam_lock:
                self.cam.AcquisitionFrameRate.SetValue(
                    min(self.cam.AcquisitionFrameRate.GetMax(), self._framerate)
                )
            self._framerate = float(self.cam.AcquisitionFrameRate.GetValue())

    @property
    def binning(self):
        if self.cam_open and self.cam is not None:
            return (self.cam.BinningHorizontal.GetValue(), self.cam.BinningVertical.GetValue())
        return (-1, -1)
    @binning.setter
    def binning(self, val) -> None:
        if val is None or (not self.cam_open) or (self.cam is None):
            return
        if isinstance(val, (tuple, list)) and len(val) > 1:
            bh = min(max(val[0], self.cam.BinningHorizontal.GetMin()), self.cam.BinningHorizontal.GetMax())
            bv = min(max(val[1], self.cam.BinningVertical.GetMin()), self.cam.BinningVertical.GetMax())
        else:
            bh = min(max(val, self.cam.BinningHorizontal.GetMin()), self.cam.BinningHorizontal.GetMax())
            bv = min(max(val, self.cam.BinningVertical.GetMin()), self.cam.BinningVertical.GetMax())
        if self._is_writable(self.cam.BinningHorizontal):
            with self.cam_lock:
                self.cam.BinningHorizontal.SetValue(int(bh))
        if self._is_writable(self.cam.BinningVertical):
            with self.cam_lock:
                self.cam.BinningVertical.SetValue(int(bv))
        self._binning = (self.cam.BinningHorizontal.GetValue(), self.cam.BinningVertical.GetValue())

    @property
    def offset(self):
        if self.cam_open and self.cam is not None:
            return (self.cam.OffsetX.GetValue(), self.cam.OffsetY.GetValue())
        return (float("nan"), float("nan"))
    @offset.setter
    def offset(self, val) -> None:
        if val is None or (not self.cam_open) or (self.cam is None):
            return
        if isinstance(val, (tuple, list)) and len(val) > 1:
            ox = min(max(val[0], self.cam.OffsetX.GetMin()), self.cam.OffsetX.GetMax())
            oy = min(max(val[1], self.cam.OffsetY.GetMin()), self.cam.OffsetY.GetMax())
        else:
            ox = min(max(val, self.cam.OffsetX.GetMin()), self.cam.OffsetX.GetMax())
            oy = min(max(val, self.cam.OffsetY.GetMin()), self.cam.OffsetY.GetMax())
        if self._is_writable(self.cam.OffsetX):
            with self.cam_lock:
                self.cam.OffsetX.SetValue(int(ox))
        if self._is_writable(self.cam.OffsetY):
            with self.cam_lock:
                self.cam.OffsetY.SetValue(int(oy))
        self._offset = (self.cam.OffsetX.GetValue(), self.cam.OffsetY.GetValue())

    @property
    def adc(self):
        if not self.cam_open or self.cam is None:
            return -1
        try:
            adc_val = self.cam.AdcBitDepth.GetValue()
            for bits, enum_name in self._ADC_ENUM.items():
                enum_val = self._getattr_enum(enum_name)
                if enum_val is not None and adc_val == enum_val:
                    return bits
        except Exception:
            pass
        return -1
    @adc.setter
    def adc(self, val) -> None:
        if (val is None) or (val == -1):
            return
        try:
            bits = int(val)
        except Exception:
            return
        self._cfg_adc_present = True
        self._adc = bits
        try:
            self._configs["adc"] = bits
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        enum_name = self._ADC_ENUM.get(bits)
        enum_val = self._getattr_enum(enum_name) if enum_name else None
        if enum_val is not None and self._is_writable(self.cam.AdcBitDepth):
            with self.cam_lock:
                self.cam.AdcBitDepth.SetValue(enum_val)
            self._adc = bits
            # Maintain legacy behavior: 8-bit -> Mono8, otherwise Mono16.
            if self._adc == 8:
                self.pixel_format = "Mono8"
            else:
                self.pixel_format = "Mono16"

    @property
    def pixel_format(self) -> str:
        if not self.cam_open or self.cam is None:
            return "unknown"
        try:
            return self._pixel_format_to_str(self.cam.PixelFormat.GetValue())
        except Exception:
            return "unknown"
    @pixel_format.setter
    def pixel_format(self, fmt: str) -> None:
        if not fmt:
            return
        const = self._pixel_format_const(fmt)
        if const is None:
            self._log(logging.WARNING, f"PySpin:Unknown pixel format '{fmt}'")
            return
        self._pixel_format_req = str(fmt)
        try:
            self._configs["pixel_format"] = self._pixel_format_req
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        if self._is_writable(self.cam.PixelFormat):
            try:
                with self.cam_lock:
                    self.cam.PixelFormat.SetValue(const)
            except Exception as exc:
                self._log(logging.WARNING, f"PySpin:PixelFormat set failed: {exc}")

    @property
    def convert_format(self) -> str:
        return str(self._convert_format)
    @convert_format.setter
    def convert_format(self, value: str) -> None:
        if not value:
            return
        self._convert_format = str(value)
        try:
            self._configs["convert_format"] = self._convert_format
        except Exception:
            pass
        if self.cam_open:
            self._configure_conversion()
            self._allocate_buffer()

    @property
    def ttlinv(self):
        if self.cam_open and self.cam is not None and self._is_writable(self.cam.LineInverter):
            try:
                return bool(self.cam.LineInverter.GetValue())
            except Exception:
                pass
        return bool(self._ttlinv)
    @ttlinv.setter
    def ttlinv(self, val) -> None:
        if val is None:
            return
        inv = bool(val)
        self._cfg_ttlinv_present = True
        self._ttlinv = inv
        try:
            self._configs["ttlinv"] = inv
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        if self._is_writable(self.cam.LineInverter):
            with self.cam_lock:
                self.cam.LineInverter.SetValue(inv)

    @property
    def trigout(self) -> int:
        return int(self._trigout)
    @trigout.setter
    def trigout(self, val) -> None:
        if val is None:
            return
        line = int(val)
        self._cfg_trigout_present = True
        if line < 0:
            self._trigout = -1
            try:
                self._configs["trigout"] = -1
            except Exception:
                pass
            return
        self._trigout = line
        try:
            self._configs["trigout"] = line
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        selector_name = f"LineSelector_Line{line}"
        selector_val = self._getattr_enum(selector_name)
        line_mode_out = self._getattr_enum("LineMode_Output")
        line_source_exp = self._getattr_enum("LineSource_ExposureActive")
        if selector_val is None:
            self._log(logging.WARNING, f"PySpin:Line {line} not supported for trigout")
            return
        with self.cam_lock:
            if self._is_writable(self.cam.LineSelector):
                self.cam.LineSelector.SetValue(selector_val)
            if line_mode_out is not None and self._is_writable(self.cam.LineMode):
                self.cam.LineMode.SetValue(line_mode_out)
            if self._is_writable(self.cam.LineInverter):
                self.cam.LineInverter.SetValue(bool(self._ttlinv))
            if line_source_exp is not None and self._is_writable(self.cam.LineSource):
                self.cam.LineSource.SetValue(line_source_exp)

    @property
    def trigin(self) -> int:
        return int(self._trigin)
    @trigin.setter
    def trigin(self, val) -> None:
        if val is None:
            return
        line = int(val)
        self._cfg_trigin_present = True
        self._trigin = line
        try:
            self._configs["trigin"] = line
        except Exception:
            pass
        if (not self.cam_open) or (self.cam is None):
            return
        trig_selector_frame = self._getattr_enum("TriggerSelector_FrameStart")
        trig_selector_acq = self._getattr_enum("TriggerSelector_AcquisitionStart")
        trig_mode_on = self._getattr_enum("TriggerMode_On")
        trig_source_soft = self._getattr_enum("TriggerSource_Software")
        trig_overlap_ro = self._getattr_enum("TriggerOverlap_ReadOut")
        trig_overlap_off = self._getattr_enum("TriggerOverlap_Off")
        trig_act_rise = self._getattr_enum("TriggerActivation_RisingEdge")
        trig_act_fall = self._getattr_enum("TriggerActivation_FallingEdge")

        with self.cam_lock:
            if line == -1:
                if trig_selector_acq is not None and self._is_writable(self.cam.TriggerSelector):
                    self.cam.TriggerSelector.SetValue(trig_selector_acq)
                if trig_mode_on is not None and self._is_writable(self.cam.TriggerMode):
                    self.cam.TriggerMode.SetValue(trig_mode_on)
                if trig_source_soft is not None and self._is_writable(self.cam.TriggerSource):
                    self.cam.TriggerSource.SetValue(trig_source_soft)
                if trig_overlap_ro is not None and self._is_writable(self.cam.TriggerOverlap):
                    self.cam.TriggerOverlap.SetValue(trig_overlap_ro)
            else:
                source_val = self._getattr_enum(f"TriggerSource_Line{line}")
                if source_val is None:
                    self._log(logging.WARNING, f"PySpin:Line {line} not supported for trigin")
                    return
                if trig_selector_frame is not None and self._is_writable(self.cam.TriggerSelector):
                    self.cam.TriggerSelector.SetValue(trig_selector_frame)
                if trig_mode_on is not None and self._is_writable(self.cam.TriggerMode):
                    self.cam.TriggerMode.SetValue(trig_mode_on)
                if self._is_writable(self.cam.TriggerSource):
                    self.cam.TriggerSource.SetValue(source_val)
                if self._is_writable(self.cam.TriggerActivation):
                    if bool(self._ttlinv) and trig_act_fall is not None:
                        self.cam.TriggerActivation.SetValue(trig_act_fall)
                    elif trig_act_rise is not None:
                        self.cam.TriggerActivation.SetValue(trig_act_rise)
                if trig_overlap_off is not None and self._is_writable(self.cam.TriggerOverlap):
                    self.cam.TriggerOverlap.SetValue(trig_overlap_off)
                if self._is_writable(self.cam.TriggerDelay):
                    self.cam.TriggerDelay.SetValue(self.cam.TriggerDelay.GetMin())

    # ------------------------------------------------------------------
    # Controls helpers
    # ------------------------------------------------------------------

    def get_control(self, name: str):
        key = str(name or "").strip()
        mapping = {
            "ExposureTime": self.exposure,
            "Exposure": self.exposure,
            "AutoExposure": self.autoexposure,
            "FrameRate": self.fps,
            "FPS": self.fps,
            "Binning": self.binning,
            "Offset": self.offset,
            "ADC": self.adc,
            "PixelFormat": self.pixel_format,
            "TriggerIn": self.trigin,
            "TriggerOut": self.trigout,
            "TTLInverted": self.ttlinv,
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
                elif k in ("AutoExposure",):
                    self.autoexposure = val
                elif k in ("FrameRate", "FPS"):
                    self.fps = val
                elif k in ("Binning",):
                    self.binning = val
                elif k in ("Offset",):
                    self.offset = val
                elif k in ("ADC",):
                    self.adc = val
                elif k in ("PixelFormat",):
                    self.pixel_format = str(val)
                elif k in ("TriggerIn",):
                    self.trigin = val
                elif k in ("TriggerOut",):
                    self.trigout = val
                elif k in ("TTLInverted",):
                    self.ttlinv = val
                elif k in ("Flip",):
                    self.flip = val
                elif k in ("Resolution", "Size"):
                    self.size = val
                else:
                    continue
                ok_any = True
            except Exception:
                continue
        return ok_any

    def get_supported_main_color_formats(self) -> list[str]:
        formats = {"native", "Mono8", "Mono16", "BGR8", "RGB8"}
        return sorted(formats)

    def get_supported_raw_color_formats(self) -> list[str]:
        return ["native", "Mono16"]

    def get_supported_main_options(self):
        return [
            {
                "size": (self._capture_width, self._capture_height),
                "pixel_format": self.pixel_format,
                "adc": self._adc,
                "fps": self._framerate,
                "convert_format": self._convert_format,
            }
        ]

    def get_supported_raw_options(self):
        return [
            {
                "size": (self._capture_width, self._capture_height),
                "pixel_format": self.pixel_format,
                "adc": self._adc,
            }
        ]

    def log_stream_options(self) -> None:
        try:
            supported = self.get_supported_main_color_formats()
            self._log(logging.INFO, f"PySpin:Supported formats:{supported}")
        except Exception:
            pass
        try:
            main_opts = self.get_supported_main_options()
            self._log(logging.INFO, f"PySpin:Main options:{main_opts}")
        except Exception:
            pass


__all__ = ["SpinCore"]
