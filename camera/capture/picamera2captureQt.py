###############################################################################
# Raspberry Pi CSI video capture (Picamera2 / libcamera)
# for Qt5/Qt6 environments
#
# Uses the picamera2 library to interface with the Raspberry Pi Camera Module
#
# Urs Utzinger
# GPT-5.2
#
# 2026 First Release
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: piCamera2CaptureQt(QObject)
#
# Threaded capture wrapper around `PiCamera2Core` for Qt.
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
#     Emitted roughly every ~5 seconds.
# - log(level: int, message: str)
#     Logging/event messages (uses Python logging levels).
# - opened()
#     Emitted when the camera has been opened/configured (also after reconfigure).
# - started()
# - stopped()
#
# Public methods:
# - open_cam(timeout: float | None = 2.0) -> bool: start camera loop thread and open/configure camera
# - close_cam(timeout: float | None = 2.0) -> None: stop camera loop thread and close camera
# - start() / stop(): enable/disable capturing (loop keeps running)
# - join(timeout: float | None = None) -> None: wait for camera loop thread exit
# - log_stream_options(): emits stream options via `log` signal (delegated to core)
# - set_exposure_us(exposure_us: int)
# - set_auto_exposure(enabled: bool)
# - set_framerate(fps: float)
# - set_aemeteringmode(mode: int|str)
# - set_auto_wb(enabled: bool)
# - set_awbmode(mode: int|str)
# - set_flip(flip: int)  # recorded; triggers internal reconfigure
# - set_resolution(res: tuple[int,int])  # recorded; triggers internal reconfigure
# - get_supported_main_formats() -> list[str]
# - get_supported_raw_formats() -> list[str]
# - get_supported_raw_options() -> list[dict]
# - get_supported_main_options() -> list[dict]
# - get_control(name: str) -> Any
# - __getattr__: delegates unknown attributes/properties to the core, so most
#   `PiCamera2Core` helpers and properties are accessible here.
# - convertQimage(frame: np.ndarray) -> QImage | None   (expects OpenCV BGR order)
#
# Frame delivery model (NO queue semantics, NO frame signal):
# - Producers push frames into `buffer` without blocking.
# - Consumers poll (typically via a QTimer in the GUI thread):
#       if camera.buffer and camera.buffer.avail > 0:
#           frame, ts_ms = camera.buffer.pull(copy=True)
#
# Reconfigure behavior:
# - `set_flip()` and `set_resolution()` request a reconfigure handled inside the camera loop.
# - If capture is running, the loop will temporarily pause capture, apply changes, then resume.
# - Callers do not need to manually stop/start.
#
# Convenience properties:
# - cam_open: bool
#
# Supported Config Parameters (configs dict)
# ------------------------------------------
# See picamera2core.py for full list.
#
###############################################################################

###############################################################################
# Imports
###############################################################################

from __future__ import annotations
import time
import logging
import threading
from queue import Queue, Empty
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .picamera2core import PiCamera2Core
from .framebuffer import FrameBuffer

try:
    from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot  # type: ignore
    from PyQt6.QtGui import QImage  # type: ignore
    _QT_API = "PyQt6"
except Exception:  # pragma: no cover
    from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot  # type: ignore
    from PyQt5.QtGui import QImage  # type: ignore
    _QT_API = "PyQt5"


# QImage format enums moved between Qt5 and Qt6
if _QT_API == "PyQt6":
    _QIMAGE_FMT_GRAY8  = QImage.Format.Format_Grayscale8
    _QIMAGE_FMT_RGB888 = QImage.Format.Format_RGB888
else:
    _QIMAGE_FMT_GRAY8  = QImage.Format_Grayscale8
    _QIMAGE_FMT_RGB888 = QImage.Format_RGB888

class piCamera2CaptureQt(QObject):
    """Qt/threading wrapper around PiCamera2Core.

    - Captures into a SPSC ring buffer: self.buffer (alias: self.capture)
    - Emits Qt signals for stats, log, started, stopped.
    - stop()/start() control capturing (camera loop remains running).

    Most camera controls and helpers are implemented by `PiCamera2Core` and are
    accessible via attribute delegation.

    Thread safety: The buffer is accessed from both the capture thread and potentially the GUI thread.
    Consumers should always use buffer.pull(copy=True) to avoid race conditions.

    After any direct property change (e.g., self._core.size = ...), always update both self.buffer and self.capture to self._core.buffer.
    """

    stats      = pyqtSignal(float)         # measured fps
    log        = pyqtSignal(int, str)      # logging level, message
    opened     = pyqtSignal()              # camera opened/configured
    started    = pyqtSignal()              # capture started
    stopped    = pyqtSignal()              # capture stopped

    def __init__(
        self, 
        configs: dict, 
        camera_num: int = 0, 
        res: tuple | None = None,
        exposure: float | None = None,
        queue_size: int = 32,
        parent: QObject | None = None
    ):

        super().__init__(parent)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}
        self._camera_num = int(camera_num)

        # Preserve constructor overrides so open/re-open uses the same intent.
        self._res_override = res
        self._exposure_override = exposure

        # Normalize configs: keep keys consistent with piCamera2Capture
        try:
            mode = str(self._configs.get("mode", "main")).lower()
            base_res = None
            if res is not None:
                base_res = tuple(res)
            else:
                base_res = tuple(self._configs.get("camera_res", (640, 480)))
            if not (isinstance(base_res, (list, tuple)) and len(base_res) >= 2):
                base_res = (640, 480)
            self._configs.setdefault("camera_res", tuple(base_res))
            self._configs.setdefault("mode", mode)
            self._configs.setdefault("main_format", str(self._configs.get("main_format", self._configs.get("format", "BGR3"))))
            self._configs.setdefault("raw_format", str(self._configs.get("raw_format", self._configs.get("format", "SRGGB8"))))
        except Exception:
            pass

        if exposure is not None:
            self._configs["exposure"] = exposure

        # Core configuration
        self._fps = float(self._configs.get("fps", 30.0))
        # Note: default is unthrottled emission. If a UI wants to display at a
        # lower rate, it should throttle in the consumer (e.g. via a QTimer).
        # Legacy/ignored: emission is unthrottled; keep for config compatibility.
        self._displayfps = float(self._configs.get("displayfps", 0))
        self._low_latency = bool(self._configs.get("low_latency", False))

        # FrameBuffer sizing (similar semantics to non-Qt wrapper)
        cfg_buffersize = self._configs.get("buffersize", None)
        if cfg_buffersize is not None:
            buffer_capacity = int(cfg_buffersize)
        elif self._low_latency:
            buffer_capacity = 1
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

        # Core exists for the lifetime of this wrapper. The actual camera is
        # opened/configured inside the capture thread.
        self._core_log_q: "Queue[tuple[int, str]]" = Queue(maxsize=32)
        self._core = PiCamera2Core(
            self._configs,
            camera_num=self._camera_num,
            res=self._res_override,
            exposure=self._exposure_override,
            log_queue=self._core_log_q,
        )

        # Camera loop lifecycle
        self._capture_thread: threading.Thread | None = None
        self._loop_stop_evt     = threading.Event() # stops capture loop
        self._capture_evt       = threading.Event() # toggles capturing on/off
        self._reconfigure_evt   = threading.Event() # request reopen/reconfigure while loop runs
        self._open_finished_evt = threading.Event() # signals camera open attempt finished

        # Open the camera immediately with the provided configs
        self._core.open_cam()

        if self._core.buffer is None:
            try:
                self.log.put_nowait((logging.ERROR, "PiCam2:Core buffer is None after open"))
            except Exception:
                pass

        self.buffer = self._core.buffer
        self.capture = self.buffer  # alias

        # Logger thread: created and started in init, runs for object lifetime
        self._logger_thread = threading.Thread(target=self._logger_loop, daemon=True)
        self._logger_thread.start()


    # ------------------------------------------------------------------
    # Lifecycle (unified with Qt wrapper semantics)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def start(self) -> bool:
        """Start the capture loop (does not open camera)."""
        if not self._core.cam_open:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, "PiCam2:Camera not open; cannot start capture"))
            except Exception:
                pass
            return False

        # Check buffer exists
        if self._core.buffer is None:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, "PiCam2:Buffer not allocated; cannot start capture"))
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
        """Stop the capture loop (does not close camera)."""
        self._capture_evt.clear()
        self._loop_stop_evt.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        self._capture_thread = None

    def pause(self):
        """Convenience: Pause capture without stopping loop."""
        self._capture_evt.clear()
        
    def resume(self):
        """Convenience: Resume capture."""
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_evt.set()
   
    @property
    def is_capturing(self) -> bool:
        """Check if actively capturing frames."""
        return self._capture_evt.is_set()
    
    @property
    def is_running(self) -> bool:
        """Check if capture loop thread is running."""
        return self._capture_thread is not None and self._capture_thread.is_alive()

    def join(self, timeout: float | None = None):
        """Block until the capture and logger threads exit."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        if self._logger_thread is not None and self._logger_thread.is_alive():
            self._logger_thread.join(timeout=timeout)

    def __del__(self):
        # Close camera and stop logger thread on object deletion
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Explicitly close camera and stop threads. Use in long-running apps for deterministic cleanup."""
        self._loop_stop_evt.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._logger_thread is not None and self._logger_thread.is_alive():
            self._logger_thread.join(timeout=2.0)
        self._core.close_cam()

    # ------------------------------------------------------------------
    # Camera control helpers (Qt/non-Qt parity)
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def set_exposure_us(self, exposure_us: int) -> None:
        """Set manual exposure in microseconds (disables AE)."""
        try:
            self._core.set_controls({"AeEnable": False, "ExposureTime": int(exposure_us)})
        except Exception as exc:
            import sys
            print(f"[LOG-ERROR] set_exposure_us signal emission failed: {exc}", file=sys.stderr)

    @pyqtSlot(bool)
    def set_auto_exposure(self, enabled: bool) -> None:
        """Enable/disable auto-exposure."""
        try:
            self._core.set_controls({"AeEnable": bool(enabled)})
        except Exception as exc:
            import sys
            print(f"[LOG-ERROR] set_auto_exposure signal emission failed: {exc}", file=sys.stderr)

    @pyqtSlot(int)
    @pyqtSlot(str)
    def set_aemeteringmode(self, mode) -> None:
        """Set AE metering mode (int or friendly string)."""
        try:
            meter_val = self._core._parse_aemeteringmode(mode)
            self._core.set_controls({"AeMeteringMode": int(meter_val)})
        except Exception as exc:
            import sys
            print(f"[LOG-ERROR] set_aemeteringmode signal emission failed: {exc}", file=sys.stderr)

    @pyqtSlot(int)
    @pyqtSlot(str)
    def set_awbmode(self, mode) -> None:
        """Set AWB mode (int or friendly string)."""
        try:
            awb_val = self._core._parse_awbmode(mode)
            self._core.set_controls({"AwbMode": int(awb_val)})
        except Exception as exc:
            import sys
            print(f"[LOG-ERROR] set_awbmode signal emission failed: {exc}", file=sys.stderr)

    @pyqtSlot(bool)
    def set_auto_wb(self, enabled: bool) -> None:
        """Enable/disable auto-white-balance."""
        try:
            self._core.set_controls({"AwbEnable": bool(enabled)})
        except Exception as exc:
            import sys
            print(f"[LOG-ERROR] set_auto_wb signal emission failed: {exc}", file=sys.stderr)

    def set_framerate(self, fps: float) -> None:
        """Set requested capture framerate (applies live)."""
        try:
            fr = float(fps)
            if fr > 0:
                frame_us = int(round(1_000_000.0 / fr))
                if frame_us > 0:
                    self._core.set_controls({"FrameDurationLimits": (frame_us, frame_us)})
                    return
            # Best-effort fallback
            self._core.set_controls({"FrameRate": float(fr)})
        except Exception as exc:
            import sys
            print(f"[LOG-ERROR] set_framerate signal emission failed: {exc}", file=sys.stderr)

    @pyqtSlot(int)
    def set_flip(self, flip: int) -> None:
        """Record a new flip setting.

        Flip changes trigger an internal reconfigure. If capture is running,
        the camera loop will temporarily pause capture, apply the new flip, and
        resume capture automatically. If not running, the change is applied immediately.
        After any change, both self.buffer and self.capture are updated to self._core.buffer.
        """
        try:
            f = int(flip)
            self._configs["flip"] = f

            # If loop is running, signal reconfigure
            if self._capture_thread and self._capture_thread.is_alive():
                self._pending_flip = f
                self._reconfigure_evt.set()
                try:
                    self.log.emit(logging.INFO, f"PiCam2:flip set to {f}; will reconfigure")
                except Exception:
                    pass
            else:
                # Not running - just apply directly
                try:
                    self._core.flip = f
                    self.buffer = self._core.buffer
                    self.capture = self.buffer
                    self.log.emit(logging.INFO, f"PiCam2:flip set to {f}")
                except Exception as exc:
                    self.log.emit(logging.ERROR, f"PiCam2:Failed to set flip: {exc}")
        except Exception:
            pass

    @pyqtSlot(tuple)
    def set_resolution(self, res: Union[tuple, list]) -> None:
        """Record a new capture resolution.

        Resolution changes trigger an internal reconfigure. If capture is
        running, the camera loop will temporarily pause capture, apply the new
        size, and resume capture automatically. If not running, the change is applied immediately.
        After any change, both self.buffer and self.capture are updated to self._core.buffer.
        """
        try:
            w, h = int(res[0]), int(res[1])
            if w <= 0 or h <= 0:
                return

            #  Record request;
            self._configs["camera_res"] = (w, h)
            
            # If loop is running, signal reconfigure
            if self._capture_thread and self._capture_thread.is_alive():
                self._pending_camera_res = (w, h)
                self._reconfigure_evt.set()
                self.log.emit(logging.INFO, f"PiCam2:resolution set to {w}x{h}; will reconfigure")
            else:
                # Not running - just apply directly
                try:
                    self._core.size = (w, h)
                    self.buffer = self._core.buffer
                    self.capture = self.buffer
                    self.log.emit(logging.INFO, f"PiCam2:resolution set to {w}x{h}")
                except Exception as exc:
                    self.log.emit(logging.ERROR, f"PiCam2:Failed to set resolution: {exc}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read-only helpers (safe for GUI thread; best-effort)
    # ------------------------------------------------------------------

    def get_supported_main_formats(self) -> list[str]:
        """Return common main-stream pixel formats (libcamera/Picamera2 naming).

        Kept for API parity with the non-Qt wrapper. This delegates to the core's
        `get_supported_main_color_formats()`.
        """
        try:
            return list(self._core.get_supported_main_color_formats())
        except Exception:
            # Fallback (mirrors core list; kept here so GUI code can call without start()).
            return ["XBGR8888", "XRGB8888", "RGB888", "BGR888", "YUV420", "YUYV", "MJPEG"]

    def get_supported_raw_formats(self) -> list[str]:
        """Return supported RAW Bayer format strings (best-effort)."""
        try:
            return list(self._core.get_supported_raw_color_formats())
        except Exception:
            return []

    def get_supported_raw_options(self):
        """Return available RAW (Bayer) sensor modes.

        Returns a list of dicts: {"format": str, "size": (w, h), "fps": float, "area": int}
        """
        try:
            return list(self._core.get_supported_raw_options())
        except Exception:
            return []

    def get_supported_main_options(self):
        """Best-effort suggested main-stream choices based on sensor modes."""
        try:
            return list(self._core.get_supported_main_options())
        except Exception:
            return []

    def log_stream_options(self) -> None:
        """Emit helpful camera stream options via the `log` signal.

        This mirrors the informational output used by the threaded wrapper/core.
        It is best-effort and requires an opened camera (`start()` must have run).
        """
        try:
            self._core.log_stream_options()
        except Exception:
            pass

    def log_camera_config_and_controls(self) -> None:
        """Emit current camera configuration and controls via the `log` signal."""
        try:
            self._core.log_camera_config_and_controls()
        except Exception:
            pass

    def get_control(self, name: str):
        try:
            return self._core.get_control(name)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal camera loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:

        # Local bindings (hot loop) - mirror non-Qt wrapper structure
        loop_stop_evt        = self._loop_stop_evt
        capture_evt          = self._capture_evt
        reconfigure_evt      = self._reconfigure_evt
        core                 = self._core

        last_drop_warn_t     = 0.0
        fps_measurement_interval = 5.0  # Measure FPS every 5 seconds
        fps_frame_count      = 0
        fps_start_time       = time.perf_counter()

        # Track capture state transitions for started/stopped signals
        capturing_prev = False

        fb                   = self._core.buffer
        if fb is None:
            try:
                self.log.emit(logging.CRITICAL, "PiCam2:No buffer allocated")
            except Exception:
                pass
            return

        try:
            while not loop_stop_evt.is_set():

                # Update Capturing State
                # ----------------------------------------------------------------------------------------

                capturing_now = bool(capture_evt.is_set())
                if capturing_now != capturing_prev:
                    capturing_prev = capturing_now
                    # Reset FPS measurement on state change
                    fps_frame_count = 0
                    fps_start_time = time.perf_counter()

                    if capturing_now:
                        self.started.emit()
                    else:
                        self.stopped.emit()

                # Capture Frame
                # ----------------------------------------------------------------------------------------

                if capturing_now:
                    loop_start = time.perf_counter()

                    # drain captured frames
                    try:
                        frame, ts_ms = core.capture_array()
                    except Exception as exc:
                        try:
                            self.log.emit(logging.WARNING, f"PiCam2:capture_array failed: {exc}")
                        except Exception:
                            pass
                        time.sleep(0.001)
                        continue

                    if frame is None:
                        time.sleep(0.001)
                        continue

                    # time stamp
                    frame_time = float(ts_ms if ts_ms is not None else (loop_start * 1000.0))

                    # Push into FrameBuffer without blocking
                    # --------------------------------------
                    try:
                        ok_push = bool(fb.push(frame, frame_time))

                        if (not ok_push):
                            now = time.perf_counter()
                            if (now - last_drop_warn_t) >= 1.0:
                                last_drop_warn_t = now
                                try:
                                    self.log.emit(logging.WARNING, "PiCam2:FrameBuffer is full; dropping frame")
                                except RuntimeError:
                                    pass
                        else:
                            # Successful push - update FPS measurement
                            fps_frame_count += 1

                    except Exception as exc1:
                        # Most likely a shape/dtype mismatch due to reconfigure.
                        try:
                            self.log.emit(logging.WARNING, f"PiCam2:FrameBuffer push failed ({exc1}); reallocating")
                        except Exception:
                            pass

                        try:

                            # Determine new buffer shape from frame
                            h, w = frame.shape[:2]
                            c = frame.shape[2] if frame.ndim == 3 else 1
                            new_shape = (h, w, c) if c > 1 else (h, w)
                            
                            # Reallocate buffer with new shape
                            core._buffer = FrameBuffer(
                                capacity=self._buffer_capacity,
                                frame_shape=new_shape,
                                dtype=frame.dtype,
                                overwrite=self._buffer_overwrite
                            )
                            
                            # Update all references immediately
                            self.buffer = core.buffer
                            self.capture = self.buffer
                            fb = core.buffer
                            
                            # Retry push
                            ok_push = fb.push(frame, frame_time)
                            if not ok_push:
                                raise RuntimeError("FrameBuffer still full after reallocation")

                            # Reset FPS measurement after buffer change
                            fps_frame_count = 0
                            fps_start_time = time.perf_counter()

                        except Exception as exc2:
                                try:
                                    self.log.emit(logging.CRITICAL, 
                                        f"PiCam2:FrameBuffer retry failed ({exc2}); stopping")
                                except Exception:
                                    pass
                                loop_stop_evt.set()
                                break

                    # Measure performance
                    # -------------------
                    elapsed = loop_start - fps_start_time
                    if elapsed >= fps_measurement_interval:
                        self._measured_fps = fps_frame_count / elapsed
                        fps_start_time = loop_start
                        fps_frame_count = 0

                else:
                    self._measured_fps = 0.0
                    # Idle: keep loop responsive without busy-spinning
                    time.sleep(0.01)

                # End Capture Frame ------------------------------------------------------------------------


                # Apply reconfigure request (auto-stop capture -> apply -> auto-restart)
                # ------------------------------------------------------------------------------------------
                if reconfigure_evt.is_set():
                    was_capturing = bool(capture_evt.is_set())

                    # Force capture off so we can safely reconfigure.
                    capture_evt.clear()

                    try:
                        reconfigure_evt.clear()

                        # Apply pending size change (core will restart camera if open).
                        if self._pending_camera_res is not None:
                            try:
                                w, h = self._pending_camera_res
                                core.size = (w, h)
                                try:
                                    logq.put_nowait((logging.INFO, f"PiCam2:Reconfigured resolution to {w}x{h}"))
                                except Exception:
                                    pass
                            except Exception as exc:
                                try:
                                    logq.put_nowait((logging.ERROR, f"PiCam2:Size change failed: {exc}"))
                                except Exception:
                                    pass
                            finally:
                                self._pending_camera_res = None

                        # Apply pending flip change (core will restart camera if open).
                        if self._pending_flip is not None:
                            try:
                                f = self._pending_flip
                                core.flip = f
                                try:
                                    logq.put_nowait((logging.INFO, f"PiCam2:Reconfigured flip to {f}"))
                                except Exception:
                                    pass
                            except Exception as exc:
                                try:
                                    logq.put_nowait((logging.ERROR, f"PiCam2:Flip change failed: {exc}"))
                                except Exception:
                                    pass
                            finally:
                                self._pending_flip = None
                                

                        # Verify camera is still open
                        if not core.cam_open:
                            raise RuntimeError("camera not open after reconfigure")

                        # Reset/reallocate buffer after configuration change.

                        if self.buffer is not None:
                            try:
                                self.buffer.clear()
                            except Exception:
                                pass
                        
                        self.buffer = core.buffer
                        self.capture = self.buffer
                        fb = core.buffer

                        # Reset FPS measurement
                        fps_frame_count = 0
                        fps_start_time = time.perf_counter()

                    except Exception as exc:
                        try:
                            self.log.emit(logging.CRITICAL, f"PiCam2:Reconfigure failed: {exc}")
                        except Exception:
                            pass
                        loop_stop_evt.set()
                        break

                    # Resume capture if it was previously running.
                    if was_capturing:
                        capture_evt.set()
                    continue

                # End Apply reconfigure ---------------------------------------------------------------------

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
        """Periodically drain logs and emit them via Qt signals."""
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
            time.sleep(0.2)  # Adjust as needed for log responsiveness

    def log_camera_config_and_controls(self) -> None:
        """Log current camera configuration and key timing/AE controls."""
        try:
            fn = getattr(self._core, "log_camera_config_and_controls", None)
            if callable(fn):
                fn()
                return
        except Exception:
            pass
        try:
            self.log.emit(logging.WARNING, "PiCam2:log_camera_config_and_controls not available")
        except Exception:
            pass

    def __getattr__(self, name: str):
        """Convenience: delegate unknown attributes to the core."""
        return getattr(self._core, name)

    @property
    def cam_open(self) -> bool:
        """Check if camera is open."""
        return self._core.cam_open
        
    @property
    def measured_fps(self) -> float:
        # This Qt class emits fps via stats(); keep a convenience getter.
        try:
            return float(self._measured_fps)
        except Exception:
            return 0.0

    @measured_fps.setter
    def measured_fps(self, value: float) -> None:
        """Set measured FPS."""
        try:
            self._measured_fps = float(value)
        except Exception:
            self._measured_fps = 0.0

    def convertQimage(self, frame) -> QImage | None:
        """Convert an OpenCV-style numpy frame to QImage for display.

        Expected input: uint8 HxWx3 in BGR (OpenCV order) or a 2D grayscale image.
        Returns a deep-copied QImage (safe after returning).
        """
        try:
            if frame is None:
                return None
            if not isinstance(frame, np.ndarray):
                # Be permissive: allow array-like inputs.
                try:
                    frame = np.asarray(frame)
                except Exception:
                    return None

            # Some pipelines can produce (H, W, 1) or (H, W, 2). Normalize those.
            if frame.ndim == 3 and frame.shape[2] == 1:
                try:
                    frame = frame[:, :, 0]
                except Exception:
                    return None
            elif frame.ndim == 3 and frame.shape[2] == 2:
                # Keep the first channel (display-only fallback).
                try:
                    frame = frame[:, :, 0]
                except Exception:
                    return None

            # Qt expects 8-bit pixels for the formats we use here.
            # Be permissive and downcast common dtypes.
            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating):
                    # Heuristic: treat [0,1] floats as normalized.
                    maxv = float(np.nanmax(frame)) if frame.size else 0.0
                    if maxv <= 1.0:
                        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
                else:
                    # For uint16/uint32/etc, keep the low 8 bits (fast, display-only).
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

            # Assume BGR input and swap to RGB for Qt.
            # Important: avoid negative-stride channel reversal views, which can
            # break QImage construction on some stacks.
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


__all__ = ["piCamera2CaptureQt"]
