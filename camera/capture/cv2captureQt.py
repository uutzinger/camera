###############################################################################
# OpenCV video capture (cv2 wrapper) for Qt5/Qt6 environments
#
# Threaded capture wrapper around cv2Core for Qt.
#
# Urs Utzinger
# GPT-5.2
#
# 2026 First Release
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: cv2CaptureQt(QObject)
#
# Threaded capture wrapper around `cv2Core` for Qt.
#
# Owns an internal camera loop thread (not a QThread subclass).
#
# Public attributes:
# - buffer: FrameBuffer
#     Single-producer/single-consumer ring buffer storing (frame, ts_ms).
#     Producer can overwrite when full (configurable).
# - capture: FrameBuffer
#     Alias of `buffer` for historical naming (note: NOT a Queue).
#
# Signals (Qt):
# - stats(measured_fps: float)
# - log(level: int, message: str)
# - opened()
# - started()
# - stopped()
#
# Public methods:
# - start() / stop(): enable/disable capturing (loop keeps running)
# - join(): wait for camera loop thread exit
# - log_stream_options(): delegated to core
# - close(timeout: float | None = 2.0): stop loop and close camera
# - set_exposure_us(exposure_us: int)
# - set_auto_exposure(enabled: bool)
# - set_framerate(fps: float)
# - set_aemeteringmode(mode: int|str)   (best-effort; not supported by OpenCV)
# - set_auto_wb(enabled: bool)
# - set_awbmode(mode: int|str)          (best-effort; not supported by OpenCV)
# - set_resolution(res: tuple[int,int])  # recorded; triggers internal reconfigure
# - set_flip(flip: int)  # recorded; triggers internal reconfigure
# - get_supported_main_formats() -> list[str]
# - get_supported_raw_formats() -> list[str]
# - get_supported_raw_options() -> list[dict]
# - get_supported_main_options() -> list[dict]
# - get_control(name: str) -> Any
# - convertQimage(frame: np.ndarray) -> QImage | None   (expects OpenCV BGR order)
#
# Convenience properties:
# - cam_open: bool
# - measured_fps: float
#
# Supported Config Parameters (configs dict)
# ------------------------------------------
# See cv2core.py for full list.
#
# Note:
# - Camera is opened in __init__ with the provided configs. start()/stop() only
#   control frame capture, not camera open/close.
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

from .cv2core import cv2Core
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


class cv2CaptureQt(QObject):
    """
    Qt/threading wrapper around cv2Core.

    Public properties:
    - cam_open: bool
        Indicates if the camera is open.
    - measured_fps: float
        Returns the measured frames per second.
    """

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

        if exposure is not None:
            self._configs["exposure"] = exposure

        # FrameBuffer sizing
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

        # Runtime stats
        self._measured_fps: float = 0.0

        # Core exists for the lifetime of this wrapper
        self._core_log_q: "Queue[tuple[int, str]]" = Queue(maxsize=32)
        self._core = cv2Core(
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

        # Open the camera immediately with the provided configs
        self._core.open_cam()

        self.buffer = self._core.buffer
        self.capture = self.buffer

        # Logger thread: created and started in init, runs for object lifetime
        self._logger_thread = threading.Thread(target=self._logger_loop, daemon=True)
        self._logger_thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @pyqtSlot()
    def start(self) -> bool:
        if not self._core.cam_open:
            try:
                self.log.emit(logging.INFO, "CV2:Camera not open; cannot start capture")
            except Exception:
                pass
            return False

        if self._core.buffer is None:
            try:
                self.log.emit(logging.ERROR, "CV2:Buffer not allocated; cannot start capture")
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
    # Camera control helpers (Qt/non-Qt parity)
    # ------------------------------------------------------------------

    def set_exposure_us(self, exposure_us: int) -> None:
        """Set manual exposure (backend-dependent units, often microseconds)."""
        try:
            self._core.exposure = float(exposure_us)
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_exposure_us failed: {exc}")
            except Exception:
                pass

    def set_auto_exposure(self, enabled: bool) -> None:
        """Enable/disable auto-exposure (best-effort)."""
        try:
            self._core.autoexposure = 1 if bool(enabled) else 0
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_auto_exposure failed: {exc}")
            except Exception:
                pass

    def set_framerate(self, fps: float) -> None:
        """Set requested capture framerate (best-effort)."""
        try:
            self._core.fps = float(fps)
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_framerate failed: {exc}")
            except Exception:
                pass

    def set_aemeteringmode(self, mode) -> None:
        """Set AE metering mode (not supported by OpenCV)."""
        try:
            self.log.emit(logging.WARNING, "set_aemeteringmode not supported by OpenCV backends")
        except Exception:
            pass

    def set_auto_wb(self, enabled: bool) -> None:
        """Enable/disable auto-white-balance (best-effort)."""
        try:
            self._core.autowb = 1 if bool(enabled) else 0
        except Exception as exc:
            try:
                self.log.emit(logging.ERROR, f"set_auto_wb failed: {exc}")
            except Exception:
                pass

    def set_awbmode(self, mode) -> None:
        """Set AWB mode (not supported by OpenCV)."""
        try:
            self.log.emit(logging.WARNING, "set_awbmode not supported by OpenCV backends")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reconfigure
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
                self.log.emit(logging.CRITICAL, "CV2:No buffer allocated")
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
                            self.log.emit(logging.WARNING, f"CV2:capture_array failed: {exc}")
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
                                    self.log.emit(logging.WARNING, "CV2:FrameBuffer is full; dropping frame")
                                except RuntimeError:
                                    pass
                        else:
                            fps_frame_count += 1

                    except Exception as exc1:
                        try:
                            self.log.emit(logging.WARNING, f"CV2:FrameBuffer push failed ({exc1}); reallocating")
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
                                self.log.emit(logging.CRITICAL, f"CV2:FrameBuffer retry failed ({exc2}); stopping")
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

                        if self._pending_camera_res is not None:
                            try:
                                w, h = self._pending_camera_res
                                core.size = (w, h)
                                try:
                                    self.log.emit(logging.INFO, f"CV2:Reconfigured resolution to {w}x{h}")
                                except Exception:
                                    pass
                            except Exception as exc:
                                try:
                                    self.log.emit(logging.ERROR, f"CV2:Size change failed: {exc}")
                                except Exception:
                                    pass
                            finally:
                                self._pending_camera_res = None

                        if self._pending_flip is not None:
                            try:
                                f = self._pending_flip
                                core.flip = f
                                try:
                                    self.log.emit(logging.INFO, f"CV2:Reconfigured flip to {f}")
                                except Exception:
                                    pass
                            except Exception as exc:
                                try:
                                    self.log.emit(logging.ERROR, f"CV2:Flip change failed: {exc}")
                                except Exception:
                                    pass
                            finally:
                                self._pending_flip = None

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
                            self.log.emit(logging.CRITICAL, f"CV2:Reconfigure failed: {exc}")
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
        try:
            if frame is None:
                return None
            if not isinstance(frame, np.ndarray):
                try:
                    frame = np.asarray(frame)
                except Exception:
                    return None

            if frame.ndim == 3 and frame.shape[2] == 1:
                try:
                    frame = frame[:, :, 0]
                except Exception:
                    return None
            elif frame.ndim == 3 and frame.shape[2] == 2:
                try:
                    frame = frame[:, :, 0]
                except Exception:
                    return None

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
                return QImage(
                    frame.data,
                    w,
                    h,
                    int(frame.strides[0]),
                    _QIMAGE_FMT_GRAY8,
                ).copy()

            if frame.ndim != 3 or frame.shape[2] < 3:
                return None

            bgr = np.ascontiguousarray(frame[:, :, :3])
            rgb = np.ascontiguousarray(bgr[:, :, ::-1])
            h, w, _ = rgb.shape
            return QImage(
                rgb.data,
                w,
                h,
                int(rgb.strides[0]),
                _QIMAGE_FMT_RGB888,
            ).copy()
        except Exception:
            return None


__all__ = ["cv2CaptureQt"]
