###############################################################################
# Basler pylon video capture (PyPylon / GenICam) for Qt5/Qt6 environments
#
# Threaded capture wrapper around PylonCore for Qt.
###############################################################################

###############################################################################
# Public API & Supported Controls
#
# Class: pylonCaptureQt(QObject)
#
# - Frame delivery uses FrameBuffer polling (no per-frame signal).
# - Provides parity helpers shared with other wrappers where supported.
# - Stream policy / performance controls delegate to the core.
# - Basler-specific controls (triggers/binning/ADC) delegate to the core.
#
# Parity helpers:
# - start() / stop() / pause() / resume() / close() / join()
# - set_exposure_us(), set_auto_exposure(), set_framerate()
# - set_flip(), set_resolution()
# - set_auto_wb(), set_awbmode() (map to balance_white_auto)
#
# Stream policy / performance helpers:
# - set_stream_policy(), set_isp_enable(), set_gain(), set_gain_auto()
# - set_acquisition_mode(), set_exposure_mode(), set_acquisition_frame_rate_enable()
# - set_gamma(), set_gamma_enable(), set_balance_white_auto()
#
# Basler-specific helpers:
# - set_binning(), set_offset(), set_adc()
# - set_pixel_format(), set_convert_format()
# - set_trigin(), set_trigout(), set_ttlinv(), trigger_software()
#
# Stream policy behavior (applied first, then explicit overrides from config/setters):
# - 'max_fps':
#     isp_enable=False
#     gain_auto=0, gain=1.0
#     autoexposure=0, exposure_mode='timed'
#     acquisition_mode='continuous'
#     acquisition_frame_rate_enable=True
#     gamma=1.0, gamma_enable=False
#     balance_white_auto=0
# - 'auto':
#     isp_enable=True
#     autoexposure=1, gain_auto=1, balance_white_auto=1
#     acquisition_mode='continuous'
#     gamma_enable=True
# - 'default':
#     no baseline overrides; rely on camera/driver defaults.
#
# Notes:
# - Set `set_stream_policy(...)` before fine-tuning controls.
# - Any explicitly set control (config or setter) overrides the policy baseline.
###############################################################################

from __future__ import annotations

import time
import logging
import threading
from queue import Queue, Empty
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .pyloncore import PylonCore
from .framebuffer import FrameBuffer

try:
    from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot  # type: ignore
    from PyQt6.QtGui import QImage  # type: ignore
    _QT_API = "PyQt6"
except Exception:  # pragma: no cover
    from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot  # type: ignore
    from PyQt5.QtGui import QImage  # type: ignore
    _QT_API = "PyQt5"

if _QT_API == "PyQt6":
    _QIMAGE_FMT_GRAY8 = QImage.Format.Format_Grayscale8
    _QIMAGE_FMT_RGB888 = QImage.Format.Format_RGB888
else:
    _QIMAGE_FMT_GRAY8 = QImage.Format_Grayscale8
    _QIMAGE_FMT_RGB888 = QImage.Format_RGB888


class pylonCaptureQt(QObject):
    """Qt/threading wrapper around PylonCore."""

    stats = pyqtSignal(float)
    log = pyqtSignal(int, str)
    opened = pyqtSignal()
    started = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(
        self,
        configs: dict,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        queue_size: int = 32,
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        self._configs = configs or {}
        self._camera_num = int(camera_num)
        self._res_override = res
        self._exposure_override = exposure

        base_res = res if res is not None else self._configs.get("camera_res", (720, 540))
        if isinstance(base_res, (list, tuple)) and len(base_res) >= 2:
            base_res = (int(base_res[0]), int(base_res[1]))
        else:
            base_res = (720, 540)
        self._configs.setdefault("camera_res", base_res)

        if exposure is not None:
            self._configs["exposure"] = exposure

        cfg_buffersize = self._configs.get("buffersize", None)
        if cfg_buffersize is not None:
            buffer_capacity = int(cfg_buffersize)
        else:
            buffer_capacity = int(queue_size)
        if buffer_capacity < 1:
            buffer_capacity = 1

        self._buffer_capacity = int(buffer_capacity)
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        self._buffer_copy_on_pull = bool(self._configs.get("buffer_copy", False))

        self.buffer: Optional[FrameBuffer] = None
        self.capture: Optional[FrameBuffer] = None  # alias

        # Pending reconfigure-only changes (applied by loop)
        self._pending_camera_res: tuple[int, int] | None = None
        self._pending_flip: int | None = None
        self._pending_binning = None
        self._pending_offset = None
        self._pending_adc: int | None = None
        self._pending_pixel_format: str | None = None
        self._pending_convert_format: str | None = None
        self._pending_trigin: int | None = None
        self._pending_trigout: int | None = None
        self._pending_ttlinv: bool | None = None
        self._pending_stream_policy: str | None = None
        self._pending_isp_enable: bool | None = None
        self._pending_gain: float | None = None
        self._pending_gain_auto = None
        self._pending_acquisition_mode: str | None = None
        self._pending_exposure_mode: str | None = None
        self._pending_acquisition_frame_rate_enable: bool | None = None
        self._pending_gamma: float | None = None
        self._pending_gamma_enable: bool | None = None
        self._pending_balance_white_auto = None

        # Runtime stats
        self._measured_fps: float = 0.0

        self._core_log_q: "Queue[tuple[int, str]]" = Queue(maxsize=64)
        self._core = PylonCore(
            self._configs,
            camera_num=self._camera_num,
            res=self._res_override,
            exposure=self._exposure_override,
            log_queue=self._core_log_q,
        )

        # Camera loop lifecycle
        self._capture_thread: threading.Thread | None = None
        self._loop_stop_evt = threading.Event()
        self._capture_evt = threading.Event()
        self._reconfigure_evt = threading.Event()
        self._open_finished_evt = threading.Event()

        # Open immediately
        self._core.open_cam()
        self.buffer = self._core.buffer
        self.capture = self.buffer

        self._logger_thread = threading.Thread(target=self._logger_loop, daemon=True)
        self._logger_thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @pyqtSlot()
    def start(self) -> bool:
        if not self._core.cam_open:
            try:
                self.log.emit(logging.INFO, "PyPylon:Camera not open; cannot start capture")
            except Exception:
                pass
            return False

        if self._core.buffer is None:
            try:
                self.log.emit(logging.ERROR, "PyPylon:Buffer not allocated; cannot start capture")
            except Exception:
                pass
            return False

        if self._capture_thread is None or not self._capture_thread.is_alive():
            self._loop_stop_evt.clear()
            self._capture_evt.set()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
        return True

    @pyqtSlot()
    def stop(self, timeout: float | None = 2.0):
        self._capture_evt.clear()
        self._loop_stop_evt.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        self._capture_thread = None

    def pause(self):
        self._capture_evt.clear()

    def resume(self):
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_evt.set()

    @property
    def is_capturing(self) -> bool:
        return self._capture_evt.is_set()

    @property
    def is_running(self) -> bool:
        return self._capture_thread is not None and self._capture_thread.is_alive()

    def join(self, timeout: float | None = None):
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        if self._logger_thread is not None and self._logger_thread.is_alive():
            self._logger_thread.join(timeout=timeout)

    def close(self, timeout: float | None = 2.0):
        self.stop(timeout=timeout)
        try:
            self._core.close_cam()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Parity helpers
    # ------------------------------------------------------------------

    def set_exposure_us(self, exposure_us: int) -> None:
        try:
            self._core.exposure = float(exposure_us)
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_exposure_us failed: {exc}")
            except Exception:
                pass

    def set_auto_exposure(self, enabled: bool) -> None:
        try:
            self._core.autoexposure = 1 if bool(enabled) else 0
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_auto_exposure failed: {exc}")
            except Exception:
                pass

    def set_framerate(self, fps: float) -> None:
        try:
            self._core.fps = float(fps)
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_framerate failed: {exc}")
            except Exception:
                pass

    def set_aemeteringmode(self, mode) -> None:
        try:
            self.log.emit(logging.WARNING, "set_aemeteringmode not supported by PyPylon backend")
        except Exception:
            pass

    def set_auto_wb(self, enabled: bool) -> None:
        self._pending_balance_white_auto = 1 if bool(enabled) else 0
        self._reconfigure_evt.set()

    def set_awbmode(self, mode) -> None:
        if isinstance(mode, str):
            m = mode.strip().lower()
            if m == "once":
                self._pending_balance_white_auto = "once"
            elif m in ("off", "manual", "0"):
                self._pending_balance_white_auto = 0
            else:
                self._pending_balance_white_auto = 1
        else:
            try:
                self._pending_balance_white_auto = 1 if int(mode) > 0 else 0
            except Exception:
                return
        self._reconfigure_evt.set()

    # ------------------------------------------------------------------
    # Stream policy / performance controls (reconfigure)
    # ------------------------------------------------------------------

    def set_stream_policy(self, policy: str) -> None:
        if not policy:
            return
        self._pending_stream_policy = str(policy)
        self._reconfigure_evt.set()

    def set_isp_enable(self, enabled: bool) -> None:
        self._pending_isp_enable = bool(enabled)
        self._reconfigure_evt.set()

    def set_gain(self, gain: float) -> None:
        try:
            self._pending_gain = float(gain)
        except Exception:
            return
        self._reconfigure_evt.set()

    def set_gain_auto(self, enabled_or_mode) -> None:
        self._pending_gain_auto = enabled_or_mode
        self._reconfigure_evt.set()

    def set_acquisition_mode(self, mode: str) -> None:
        if not mode:
            return
        self._pending_acquisition_mode = str(mode)
        self._reconfigure_evt.set()

    def set_exposure_mode(self, mode: str) -> None:
        if not mode:
            return
        self._pending_exposure_mode = str(mode)
        self._reconfigure_evt.set()

    def set_acquisition_frame_rate_enable(self, enabled: bool) -> None:
        self._pending_acquisition_frame_rate_enable = bool(enabled)
        self._reconfigure_evt.set()

    def set_gamma(self, value: float) -> None:
        try:
            self._pending_gamma = float(value)
        except Exception:
            return
        self._reconfigure_evt.set()

    def set_gamma_enable(self, enabled: bool) -> None:
        self._pending_gamma_enable = bool(enabled)
        self._reconfigure_evt.set()

    def set_balance_white_auto(self, enabled_or_mode) -> None:
        self._pending_balance_white_auto = enabled_or_mode
        self._reconfigure_evt.set()

    # ------------------------------------------------------------------
    # Basler-specific controls (reconfigure)
    # ------------------------------------------------------------------

    def set_resolution(self, res: tuple[int, int]) -> None:
        try:
            w, h = int(res[0]), int(res[1])
        except Exception:
            return
        self._pending_camera_res = (w, h)
        self._reconfigure_evt.set()

    def set_flip(self, flip: int) -> None:
        try:
            f = int(flip)
        except Exception:
            return
        self._pending_flip = f
        self._reconfigure_evt.set()

    def set_binning(self, val) -> None:
        self._pending_binning = val
        self._reconfigure_evt.set()

    def set_offset(self, val) -> None:
        self._pending_offset = val
        self._reconfigure_evt.set()

    def set_adc(self, bits: int) -> None:
        try:
            self._pending_adc = int(bits)
        except Exception:
            return
        self._reconfigure_evt.set()

    def set_pixel_format(self, fmt: str) -> None:
        if not fmt:
            return
        self._pending_pixel_format = str(fmt)
        self._reconfigure_evt.set()

    def set_convert_format(self, fmt: str) -> None:
        if not fmt:
            return
        self._pending_convert_format = str(fmt)
        self._reconfigure_evt.set()

    def set_trigin(self, line: int) -> None:
        try:
            self._pending_trigin = int(line)
        except Exception:
            return
        self._reconfigure_evt.set()

    def set_trigout(self, line: int) -> None:
        try:
            self._pending_trigout = int(line)
        except Exception:
            return
        self._reconfigure_evt.set()

    def set_ttlinv(self, inverted: bool) -> None:
        self._pending_ttlinv = bool(inverted)
        self._reconfigure_evt.set()

    def trigger_software(self) -> bool:
        try:
            return bool(self._core.trigger_software())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal camera loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        loop_stop_evt = self._loop_stop_evt
        capture_evt = self._capture_evt
        reconfigure_evt = self._reconfigure_evt
        core = self._core

        last_drop_warn_t = 0.0
        fps_measurement_interval = 5.0
        fps_frame_count = 0
        fps_start_time = time.perf_counter()

        capturing_prev = False

        fb = core.buffer
        if fb is None:
            try:
                self.log.emit(logging.CRITICAL, "PyPylon:No buffer allocated")
            except Exception:
                pass
            return

        try:
            while not loop_stop_evt.is_set():
                capturing_now = bool(capture_evt.is_set())
                if capturing_now != capturing_prev:
                    capturing_prev = capturing_now
                    fps_frame_count = 0
                    fps_start_time = time.perf_counter()
                    if capturing_now:
                        self.started.emit()
                    else:
                        self.stopped.emit()

                if capturing_now:
                    loop_start = time.perf_counter()
                    try:
                        frame, ts_ms = core.capture_array()
                    except Exception as exc:
                        try:
                            self.log.emit(logging.WARNING, f"PyPylon:capture_array failed: {exc}")
                        except Exception:
                            pass
                        time.sleep(0.001)
                        continue

                    if frame is None:
                        time.sleep(0.001)
                        continue

                    frame_time = float(ts_ms if ts_ms is not None else (loop_start * 1000.0))

                    try:
                        ok_push = bool(fb.push(frame, frame_time))
                        if not ok_push:
                            now = time.perf_counter()
                            if (now - last_drop_warn_t) >= 1.0:
                                last_drop_warn_t = now
                                try:
                                    self.log.emit(logging.WARNING, "PyPylon:FrameBuffer full; dropping frame")
                                except RuntimeError:
                                    pass
                        else:
                            fps_frame_count += 1
                    except Exception as exc1:
                        try:
                            self.log.emit(logging.WARNING, f"PyPylon:FrameBuffer push failed ({exc1}); reallocating")
                        except Exception:
                            pass
                        try:
                            h, w = frame.shape[:2]
                            c = frame.shape[2] if frame.ndim == 3 else 1
                            new_shape = (h, w, c) if c > 1 else (h, w)

                            core._buffer = FrameBuffer(
                                capacity=self._buffer_capacity,
                                frame_shape=new_shape,
                                dtype=frame.dtype,
                                overwrite=self._buffer_overwrite,
                            )

                            self.buffer = core.buffer
                            self.capture = self.buffer
                            fb = core.buffer

                            ok_push = fb.push(frame, frame_time)
                            if not ok_push:
                                raise RuntimeError("FrameBuffer still full after reallocation")

                            fps_frame_count = 0
                            fps_start_time = time.perf_counter()
                        except Exception as exc2:
                            try:
                                self.log.emit(logging.CRITICAL, f"PyPylon:FrameBuffer retry failed ({exc2}); stopping")
                            except Exception:
                                pass
                            loop_stop_evt.set()
                            break

                    elapsed = loop_start - fps_start_time
                    if elapsed >= fps_measurement_interval:
                        self._measured_fps = fps_frame_count / elapsed
                        try:
                            self.stats.emit(float(self._measured_fps))
                        except Exception:
                            pass
                        fps_start_time = loop_start
                        fps_frame_count = 0
                else:
                    self._measured_fps = 0.0
                    time.sleep(0.01)

                # Apply reconfigure request
                if reconfigure_evt.is_set():
                    was_capturing = bool(capture_evt.is_set())
                    capture_evt.clear()
                    try:
                        reconfigure_evt.clear()

                        if self._pending_stream_policy is not None:
                            core.stream_policy = self._pending_stream_policy
                            self._pending_stream_policy = None
                        if self._pending_isp_enable is not None:
                            core.isp_enable = self._pending_isp_enable
                            self._pending_isp_enable = None
                        if self._pending_acquisition_mode is not None:
                            core.acquisition_mode = self._pending_acquisition_mode
                            self._pending_acquisition_mode = None
                        if self._pending_exposure_mode is not None:
                            core.exposure_mode = self._pending_exposure_mode
                            self._pending_exposure_mode = None
                        if self._pending_acquisition_frame_rate_enable is not None:
                            core.acquisition_frame_rate_enable = self._pending_acquisition_frame_rate_enable
                            self._pending_acquisition_frame_rate_enable = None
                        if self._pending_gain_auto is not None:
                            core.gain_auto = self._pending_gain_auto
                            self._pending_gain_auto = None
                        if self._pending_gain is not None:
                            core.gain = self._pending_gain
                            self._pending_gain = None
                        if self._pending_balance_white_auto is not None:
                            core.balance_white_auto = self._pending_balance_white_auto
                            self._pending_balance_white_auto = None
                        if self._pending_gamma_enable is not None:
                            core.gamma_enable = self._pending_gamma_enable
                            self._pending_gamma_enable = None
                        if self._pending_gamma is not None:
                            core.gamma = self._pending_gamma
                            self._pending_gamma = None

                        if self._pending_camera_res is not None:
                            core.size = self._pending_camera_res
                            self._pending_camera_res = None
                        if self._pending_flip is not None:
                            core.flip = self._pending_flip
                            self._pending_flip = None
                        if self._pending_binning is not None:
                            core.binning = self._pending_binning
                            self._pending_binning = None
                        if self._pending_offset is not None:
                            core.offset = self._pending_offset
                            self._pending_offset = None
                        if self._pending_adc is not None:
                            core.adc = self._pending_adc
                            self._pending_adc = None
                        if self._pending_pixel_format is not None:
                            core.pixel_format = self._pending_pixel_format
                            self._pending_pixel_format = None
                        if self._pending_convert_format is not None:
                            core.convert_format = self._pending_convert_format
                            self._pending_convert_format = None
                        if self._pending_ttlinv is not None:
                            core.ttlinv = self._pending_ttlinv
                            self._pending_ttlinv = None
                        if self._pending_trigin is not None:
                            core.trigin = self._pending_trigin
                            self._pending_trigin = None
                        if self._pending_trigout is not None:
                            core.trigout = self._pending_trigout
                            self._pending_trigout = None

                        if not core.cam_open:
                            raise RuntimeError("camera not open after reconfigure")

                        if self.buffer is not None:
                            try:
                                self.buffer.clear()
                            except Exception:
                                pass

                        self.buffer = core.buffer
                        self.capture = self.buffer
                        fb = core.buffer

                        fps_frame_count = 0
                        fps_start_time = time.perf_counter()
                    except Exception as exc:
                        try:
                            self.log.emit(logging.CRITICAL, f"PyPylon:Reconfigure failed: {exc}")
                        except Exception:
                            pass
                        loop_stop_evt.set()
                        break

                    if was_capturing:
                        capture_evt.set()
                    continue

        finally:
            try:
                self._capture_evt.clear()
                self._reconfigure_evt.clear()
                core.close_cam()
            except Exception:
                pass
            finally:
                self._loop_stop_evt.set()
                self._open_finished_evt.set()
                if capturing_prev:
                    try:
                        self.stopped.emit()
                    except Exception:
                        pass

    def _logger_loop(self) -> None:
        while not self._loop_stop_evt.is_set():
            try:
                q = self._core_log_q
                while True:
                    try:
                        level, msg = q.get_nowait()
                    except Empty:
                        break
                    try:
                        self.log.emit(int(level), str(msg))
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Delegates / convenience
    # ------------------------------------------------------------------

    def log_stream_options(self) -> None:
        try:
            self._core.log_stream_options()
        except Exception:
            pass

    def get_supported_main_formats(self):
        try:
            return list(self._core.get_supported_main_color_formats())
        except Exception:
            return []

    def get_supported_raw_formats(self):
        try:
            return list(self._core.get_supported_raw_color_formats())
        except Exception:
            return []

    def get_supported_raw_options(self):
        try:
            return list(self._core.get_supported_raw_options())
        except Exception:
            return []

    def get_supported_main_options(self):
        try:
            return list(self._core.get_supported_main_options())
        except Exception:
            return []

    def get_control(self, name: str):
        try:
            return self._core.get_control(name)
        except Exception:
            return None

    def __getattr__(self, name: str):
        return getattr(self._core, name)

    @property
    def cam_open(self) -> bool:
        return self._core.cam_open

    @property
    def measured_fps(self) -> float:
        try:
            return float(self._measured_fps)
        except Exception:
            return 0.0

    @measured_fps.setter
    def measured_fps(self, value: float) -> None:
        try:
            self._measured_fps = float(value)
        except Exception:
            self._measured_fps = 0.0

    def convertQimage(self, frame) -> QImage | None:
        """Convert a numpy frame (mono or BGR) to QImage for display."""
        try:
            if frame is None:
                return None
            if not isinstance(frame, np.ndarray):
                frame = np.asarray(frame)

            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    maxv = float(np.nanmax(frame)) if frame.size else 0.0
                    if maxv <= 1.0:
                        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8, copy=False)

            if frame.ndim == 2:
                h, w = frame.shape
                frame = np.ascontiguousarray(frame)
                return QImage(frame.data, w, h, int(frame.strides[0]), _QIMAGE_FMT_GRAY8).copy()

            if frame.ndim != 3 or frame.shape[2] < 3:
                return None

            bgr = np.ascontiguousarray(frame[:, :, :3])
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            h, w, _ = rgb.shape
            return QImage(rgb.data, w, h, int(rgb.strides[0]), _QIMAGE_FMT_RGB888).copy()
        except Exception:
            return None


__all__ = ["pylonCaptureQt"]
