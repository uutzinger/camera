###############################################################################
# Raspberry Pi CSI video capture (Picamera2 / libcamera)
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
# Class: piCamera2Capture()
#
# Threaded capture wrapper around `PiCamera2Core`.
#
# Owns an internal camera loop thread (not a Thread subclass).
#
# Public attributes:
# - buffer: FrameBuffer
#     Single-producer/single-consumer ring buffer storing (frame, ts_ms).
#     Producer can overwrite when full (configurable).
# - capture: FrameBuffer
#     Alias of `buffer` for historical naming (note: NOT a Queue).
# - log: Queue[(level: int, message: str)]
#     Bounded queue of log/events using Python logging levels.
#
# Public methods:
# - start() / stop(): enable/disable capturing (loop keeps running)
# - join(): wait for camera loop thread exit
# - close(): stop camera loop thread and close camera
# - log_stream_options(max_options: int = 8): delegated to core
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
#
# Frame delivery model (NO queue semantics):
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

import threading
from queue import Queue
import queue
import logging
import time
from typing import TYPE_CHECKING, Optional, Union

from .picamera2core import PiCamera2Core, FrameBuffer

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

class piCamera2Capture:
    """
    Threaded capture wrapper around PiCamera2Core.

    - Captures into a SPSC ring buffer: self.buffer (alias: self.capture)
    - Logs into a bounded Queue: self.log
    - stop()/start() control the thread

    Most camera controls and helpers are implemented by `PiCamera2Core` and are
    accessible via attribute delegation.

    Thread safety: The buffer is accessed from both the capture thread and potentially the consumer thread.
    Consumers should always use buffer.pull(copy=True) to avoid race conditions.

    After any direct property change (e.g., self._core.size = ...), always update both self.buffer and self.capture to self._core.buffer.
    """

    def __init__(
        self,
        configs: dict,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        queue_size: int = 32,
    ):
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
        self._fps = float(self._configs.get("fps", 30))
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

        self.log = Queue(maxsize=32)

        # Runtime stats
        self.frame_time: float = 0.0  # last frame timestamp in ms (float)
        self._measured_fps: float = 0.0


        # Core handles picamera2/libcamera logic and logs into self.log
        self._core = PiCamera2Core(
            self._configs,
            camera_num=self._camera_num,
            res=self._res_override,
            exposure=self._exposure_override,
            log_queue=self.log,
        )

        # Camera loop lifecycle
        self._capture_thread: threading.Thread | None = None
        self._loop_stop_evt     = threading.Event() # stops capture loop
        self._capture_evt       = threading.Event() # toggles capturing on/off
        self._reconfigure_evt   = threading.Event() # request reopen/reconfigure while loop runs
        self._open_finished_evt = threading.Event() # signals camera open attempt finished

        # Open the camera immediately with the provided configs
        self._core.open_cam()
        self.buffer = self._core._buffer
        self.capture = self.buffer  # alias



    # ------------------------------------------------------------------
    # Lifecycle (unified with Qt wrapper semantics)
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the capture loop (does not open camera)."""
        if not self._core.cam_open:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, "PiCam2:Camera not open; cannot start capture"))
            except Exception:
                pass
            return False
        if self._capture_thread is None or not self._capture_thread.is_alive():
            self._loop_stop_evt.clear()
            self._capture_evt.set()
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
        return True

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
        """Block until the capture thread exits."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)

    def __del__(self):
        # Close camera on object deletion
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Explicitly close camera and stop threads. Use in long-running apps for deterministic cleanup."""
        self._loop_stop_evt.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._core.close_cam()

    # ------------------------------------------------------------------
    # Camera control helpers (Qt/non-Qt parity)
    # ------------------------------------------------------------------

    def set_exposure_us(self, exposure_us: int) -> None:
        """Set manual exposure in microseconds (disables AE)."""
        try:
            self._core.set_controls({"AeEnable": False, "ExposureTime": int(exposure_us)})
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"set_exposure_us failed: {exc}"))

    def set_auto_exposure(self, enabled: bool) -> None:
        """Enable/disable auto-exposure."""
        try:
            self._core.set_controls({"AeEnable": bool(enabled)})
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"set_auto_exposure failed: {exc}"))

    def set_aemeteringmode(self, mode) -> None:
        """Set AE metering mode (int or friendly string)."""
        try:
            meter_val = self._core._parse_aemeteringmode(mode)
            self._core.set_controls({"AeMeteringMode": int(meter_val)})
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"set_aemeteringmode failed: {exc}"))

    def set_awbmode(self, mode) -> None:
        """Set AWB mode (int or friendly string)."""
        try:
            awb_val = self._core._parse_awbmode(mode)
            self._core.set_controls({"AwbMode": int(awb_val)})
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"set_awbmode failed: {exc}"))

    def set_auto_wb(self, enabled: bool) -> None:
        """Enable/disable auto-white-balance."""
        try:
            self._core.set_controls({"AwbEnable": bool(enabled)})
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"set_auto_wb failed: {exc}"))

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
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"set_framerate failed: {exc}"))

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
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:flip set to {f}; will reconfigure"))
                except Exception:
                    pass
            else:
                # Not running - just apply directly
                try:
                    self._core.flip = f
                    self.buffer = self._core.buffer
                    self.capture = self.buffer
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:flip set to {f}"))
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"PiCam2:Failed to set flip: {exc}"))
        except Exception:
            pass

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
                try:
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:resolution set to {w}x{h}; will reconfigure"))
                except Exception:
                    pass
            else:
                # Not running - just apply directly
                try:
                    self._core.size = (w, h)
                    self.buffer = self._core.buffer
                    self.capture = self.buffer
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:resolution set to {w}x{h}"))
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"PiCam2:Failed to set resolution: {exc}"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read-only helpers (parity with Qt wrapper)
    # ------------------------------------------------------------------

    def get_supported_main_formats(self) -> list[str]:
        """Return common main-stream pixel formats (libcamera/Picamera2 naming).

        Kept for API parity with the Qt wrapper. This delegates to the core's
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
        """Log stream capabilities/options (delegated to core).

        This mirrors the informational output used by the threaded wrapper/core.
        It is best-effort and requires an opened camera (`start()` must have run).
        """
        try:
            self._core.log_stream_options()
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
        fb                   = self._core.buffer
        logq                 = self.log

        last_fps_t = time.perf_counter()
        num_frames = 0
        last_drop_warn_t = 0.0

        # Track capture state transitions for started/stopped signals
        capturing_prev = False

        # Track what we've told the core about capture state (update-on-change).
        core_capturing_state = False

        try:
            while not loop_stop_evt.is_set():

                # Update Capturing State
                # ----------------------------------------------------------------------------------------

                capturing_now = bool(capture_evt.is_set())
                if capturing_now != capturing_prev:
                    capturing_prev = capturing_now
                    core_capturing_state = capturing_now
                    try:
                        core.capturing = capturing_now
                    except Exception:
                        pass

                    if capturing_now:
                        # reset fps window on capture start
                        last_fps_t = time.perf_counter()
                        num_frames = 0

                # Capture Frame
                # ----------------------------------------------------------------------------------------

                if capturing_now:
                    now = time.perf_counter()
                    # drain captured frames
                    try:
                        frame, ts_ms = core.capture_array()
                    except Exception:
                        frame, ts_ms = (None, None)

                    if frame is None:
                        time.sleep(0.001)
                        continue

                    num_frames += 1
                    self.frame_time = float(ts_ms if ts_ms is not None else (now * 1000.0))

                    # Push into FrameBuffer without blocking
                    # --------------------------------------
                    try:
                        ok_push = bool(fb.push(frame, ts_ms))
                        if (not ok_push):
                            if (now - last_drop_warn_t) >= 1.0:
                                last_drop_warn_t = now
                                try:
                                    logq.put_nowait((logging.WARNING, "PiCam2:FrameBuffer is full; dropping frames"))
                                except RuntimeError:
                                    pass

                    except Exception as exc1:
                        # Most likely a shape/dtype mismatch due to reconfigure.
                        try:
                            logq.put_nowait((logging.WARNING, f"PiCam2:FrameBuffer push failed ({exc1}); reallocating"))
                        except Exception:
                            pass

                        try:
                            h, w = frame.shape[:2]
                            c = frame.shape[2] if frame.ndim == 3 else 1
                            new_shape = (h, w, c) if c > 1 else (h, w)
                            
                            from .picamera2core import FrameBuffer
                            core._buffer = FrameBuffer(
                                capacity=self._buffer_capacity,
                                frame_shape=new_shape,
                                dtype=frame.dtype,
                                overwrite=self._buffer_overwrite
                            )
                            self.buffer = core.buffer
                            self.capture = self.buffer
                            fb = core.buffer
                            
                            # Retry push
                            ok_push = fb.push(frame, ts_ms if ts_ms is not None else (now * 1000.0))
                            if not ok_push:
                                raise RuntimeError("FrameBuffer still full after reallocation")

                        except Exception as exc2:
                                try:
                                    logq.put_nowait((logging.CRITICAL, f"PiCam2:FrameBuffer retry failed ({exc2}); stopping"))
                                except Exception:
                                    pass
                                loop_stop_evt.set()
                                break

                    # Measure performance fps every 5seconds
                    # --------------------------------------

                    if (now - last_fps_t) >= 5.0:
                        self.measured_fps = num_frames / max((now - last_fps_t), 1e-6)
                        logq.put_nowait((logging.INFO, f"PiCam2:Measured capture FPS: {self.measured_fps:.1f} [Hz]"))
                        last_fps_t = now
                        num_frames = 0

                else:
                    self.measured_fps = 0.0
                    # Idle: keep loop responsive without busy-spinning
                    time.sleep(0.01)

                # End Capture Frame ------------------------------------------------------------------------


                # Apply reconfigure request (auto-stop capture -> apply -> auto-restart)
                # ------------------------------------------------------------------------------------------
                if reconfigure_evt.is_set():
                    was_capturing = bool(capture_evt.is_set())

                    # Force capture off so we can safely reconfigure.
                    capture_evt.clear()

                    # Ensure core sees an idle transition (only on change).
                    if core_capturing_state:
                        core_capturing_state = False
                        try:
                            core.capturing = False
                        except Exception:
                            pass

                    try:
                        reconfigure_evt.clear()

                        # Apply pending size change (core will restart camera if open).
                        if self._pending_camera_res is not None:
                            try:
                                core.size = self._pending_camera_res
                            finally:
                                self._pending_camera_res = None

                        # Apply pending flip change (core will restart camera if open).
                        if self._pending_flip is not None:
                            try:
                                core.flip = self._pending_flip
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

                    except Exception as exc:
                        try:
                            logq.put_nowait((logging.CRITICAL, f"PiCam2:Reconfigure failed: {exc}"))
                        except queue.Full:
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
                core.capturing = False
                self._capture_evt.clear()
                self._reconfigure_evt.clear()
                core.close_cam()
            except Exception:
                pass
            finally:
                self._loop_stop_evt.set()
                self._open_finished_evt.set()
                # No stopped signal in non-Qt wrapper

    def __getattr__(self, name: str):
        """Convenience: delegate unknown attributes to the core."""
        return getattr(self._core, name)

    @property
    def cam_open(self) -> bool:
        """Check if camera is open."""
        return self._core.cam_open
        
    @property
    def measured_fps(self) -> float:
        # convinience gett
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

    def __getattr__(self, name: str):
        """Convenience: delegate unknown attributes to the core"""
        # Only called if attribute not found on wrapper.
        return getattr(self._core, name)

__all__ = ["piCamera2Capture"]
