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
# - open_cam(timeout: float | None = 2.0) -> bool: start camera loop thread and open/configure camera
# - close_cam(): stop camera loop thread and close camera
# - start() / stop(): enable/disable capturing (loop keeps running)
# - join(): wait for camera loop thread exit
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
# - Consumers poll: `if camera.buffer and camera.buffer.avail() > 0: frame, ts = camera.buffer.pull(...)`.
#
# Reconfigure behavior:
# - `set_flip()` and `set_resolution()` request a reconfigure handled inside the camera loop.
# - If capture is running, the loop will temporarily pause capture, apply changes, then resume.
# - Callers do not need to manually stop/start.
#
# Supported config parameters (configs dict):
# -------------------------------------------
# See picamera2core.py for full list.
#
###############################################################################

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import threading
from queue import Queue, Empty
import queue
import logging
import time
from typing import TYPE_CHECKING

from .picamera2core import PiCamera2Core, FrameBuffer

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

class piCamera2Capture:
    """Threaded capture wrapper around PiCamera2Core.

    - Captures into a SPSC ring buffer: self.buffer (alias: self.capture)
    - Logs into a bounded Queue: self.log
    - stop()/start() control the thread

    Most camera controls and helpers are implemented by `PiCamera2Core` and are
    accessible via attribute delegation.
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

        # Common config fields (mirrors Qt wrapper naming where useful)
        self._low_latency = bool(self._configs.get("low_latency", False))

        # Pending reconfigure-only changes
        self._pending_camera_res: tuple[int, int] | None = None
        self._pending_flip: int | None = None
            
        # Queue handling:
        low_latency    = bool(self._low_latency)
        cfg_buffersize = self._configs.get("buffersize", None)

        if cfg_buffersize is not None:
            buffer_capacity = int(cfg_buffersize)
        elif low_latency:
            buffer_capacity = 1
        else:
            buffer_capacity = int(queue_size)
        if buffer_capacity < 1:
            buffer_capacity = 1

        self._buffer_capacity = int(buffer_capacity)
        self._buffer_overwrite = bool(self._configs.get("buffer_overwrite", True))
        self._buffer_copy_on_pull = bool(self._configs.get("buffer_copy", False))

        self.buffer: FrameBuffer | None = None
        self.capture: FrameBuffer | None = None # alias of self.buffer

        self.log = Queue(maxsize=32)

        # Runtime stats
        # Timestamp for last captured frame in milliseconds (float for sub-ms resolution).
        self.frame_time: float = 0.0
        self._measured_fps = 0.0

        # Internal loop thread state
        self._thread: threading.Thread | None = None
        self._throttled_thread: threading.Thread | None = None
        self._loop_stop_evt     = threading.Event() # stops camera loop + closes camera
        self._capture_evt       = threading.Event() # toggles capturing on/off
        self._reconfigure_evt   = threading.Event() # request reopen/reconfigure while loop runs
        self._open_finished_evt = threading.Event() # signals camera open attempt finished

        # Control updates are queued and applied in the camera loop.
        # Coalescing happens per loop iteration (last-write-wins).
        self._ctrl_q: "queue.SimpleQueue[dict]" = queue.SimpleQueue()

        # Core handles picamera2/libcamera logic and logs into self.log
        self._core = PiCamera2Core(
            self._configs,
            camera_num=self._camera_num,
            res=self._res_override,
            exposure=self._exposure_override,
            log_queue=self.log,
        )

        self.cam_lock = self._core.cam_lock

        # might need
        # # Latest frame buffer (overwrite semantics)
        # self._latest_ts = 0
        # self._latest_frame = None


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Enable capturing (opens the camera loop if needed)."""
        if not (self._thread and self._thread.is_alive()):
            try:
                self.open_cam()
            except Exception:
                logging.error("PiCam2: Failed to open camera")

        if not self.cam_open:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, "PiCam2:Camera not open; cannot start capture"))
            except Exception:
                pass
            return

        self._capture_evt.set()

    def stop(self):
        """Disable capturing while keeping the camera loop alive."""
        self._capture_evt.clear()

    def join(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is None:
            return
        t.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    @property
    def cam_open(self) -> bool:
        try:
            return bool(self._core.cam_open)
        except Exception:
            return False

    def open_cam(self, timeout: float | None = 2.0) -> bool:
        """Open/configure the camera and start the internal camera loop.

        This does not automatically start capturing; call start() to enable
        capture/emit.
        """

        if self._thread and self._thread.is_alive():
            return bool(self.cam_open)

        self._open_finished_evt.clear()
        self._capture_evt.clear()
        self._reconfigure_evt.clear()
        self._loop_stop_evt.clear()

        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        if timeout is not None:
            try:
                self._open_finished_evt.wait(timeout=float(timeout))
            except Exception:
                pass
        return bool(self.cam_open)

    def close_cam(self, timeout: float | None = 2.0) -> None:
        """Stop capturing, stop the camera loop thread, and close the camera."""
        self.stop()
        self._loop_stop_evt.set()
        self.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Unified control helpers (Qt/non-Qt parity)
    # ------------------------------------------------------------------

    def set_exposure_us(self, exposure_us: int) -> None:
        """Set manual exposure in microseconds (disables AE)."""
        try:
            self._ctrl_q.put({"AeEnable": False, "ExposureTime": int(exposure_us)})
        except Exception:
            pass

    def set_auto_exposure(self, enabled: bool) -> None:
        """Enable/disable auto-exposure."""
        try:
            self._ctrl_q.put({"AeEnable": bool(enabled)})
        except Exception:
            pass

    def set_aemeteringmode(self, mode) -> None:
        """Set AE metering mode (int or friendly string)."""
        try:
            meter_val = self._core._parse_aemeteringmode(mode)
            self._ctrl_q.put({"AeMeteringMode": int(meter_val)})
        except Exception:
            pass

    def set_awbmode(self, mode) -> None:
        """Set AWB mode (int or friendly string)."""
        try:
            awb_val = self._core._parse_awbmode(mode)
            self._ctrl_q.put({"AwbMode": int(awb_val)})
        except Exception:
            pass

    def set_auto_wb(self, enabled: bool) -> None:
        """Enable/disable auto-white-balance."""
        try:
            self._ctrl_q.put({"AwbEnable": bool(enabled)})
        except Exception:
            pass

    def set_framerate(self, fps: float) -> None:
        """Set requested capture framerate (applies live)."""
        try:
            fr = float(fps)
            if fr > 0:
                frame_us = int(round(1_000_000.0 / fr))
                if frame_us > 0:
                    self._ctrl_q.put({"FrameDurationLimits": (frame_us, frame_us)})
                    return
            # Best-effort fallback
            self._ctrl_q.put({"FrameRate": float(fr)})
        except Exception:
            pass

    def set_flip(self, flip: int) -> None:
        """Record a new flip setting.

        Flip changes trigger an internal reconfigure. If capture is running,
        the camera loop will temporarily pause capture, apply the new flip, and
        resume capture automatically.
        """
        try:
            f = int(flip)
            self._configs["flip"] = f
            self._pending_flip = f
            self._reconfigure_evt.set()
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"PiCam2:flip set to {f}; will reconfigure"))
            except Exception:
                pass
        except Exception:
            pass

    def set_resolution(self, res: tuple[int, int]) -> None:
        """Record a new capture resolution.

        Resolution changes trigger an internal reconfigure. If capture is
        running, the camera loop will temporarily pause capture, apply the new
        size, and resume capture automatically.
        """
        try:
            w, h = int(res[0]), int(res[1])
            if w <= 0 or h <= 0:
                return

            # Record request; applied by camera loop via reconfigure event.
            self._configs["camera_res"] = (w, h)
            self._pending_camera_res = (w, h)
            self._reconfigure_evt.set()
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"PiCam2:resolution set to {w}x{h}; will reconfigure"))
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read-only helpers (parity with Qt wrapper)
    # ------------------------------------------------------------------

    def get_supported_main_formats(self) -> list[str]:
        """Return common main-stream pixel formats.

        Kept for API parity with the Qt wrapper. This delegates to the core's
        `get_supported_main_color_formats()`.
        """
        try:
            return list(self._core.get_supported_main_color_formats())
        except Exception:
            return ["XBGR8888", "XRGB8888", "RGB888", "BGR888", "YUV420", "YUYV", "MJPEG"]

    def get_supported_raw_formats(self) -> list[str]:
        """Return supported RAW Bayer format strings (if available)."""
        try:
            return list(self._core.get_supported_raw_color_formats())
        except Exception:
            return []

    def get_supported_raw_options(self):
        """Return available RAW (Bayer) sensor modes (requires open camera)."""
        try:
            return self._core.get_supported_raw_options()
        except Exception:
            return []

    def get_supported_main_options(self):
        """Return best-effort main-stream mode options (requires open camera)."""
        try:
            return self._core.get_supported_main_options()
        except Exception:
            return []


    def log_stream_options(self) -> None:
        """Log stream capabilities/options (delegated to core).

        Output goes into `self.log` as (level, message) tuples.
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

    def _drain_controls(self) -> None:
        merged: dict = {}
        while True:
            try:
                d = self._ctrl_q.get_nowait()
            except Empty:
                break
            if isinstance(d, dict):
                merged.update(d)
        if not merged:
            return
        try:
            self._core.set_controls(merged)
        except Exception:
            pass

    def _drain_capture(self) -> tuple["np.ndarray | None", float | None]:
        """Capture and process a single frame.

        Returns (frame, ts_ms). The frame is converted/postprocessed to be
        OpenCV-friendly by default (mirrors Qt wrapper behavior).
        """
        try:
            # Note: PiCamera2Core handles synthetic frames internally when
            # configs['test_pattern'] is enabled. Wrappers always call
            # capture_array() to keep the pipeline identical.
            frame, ts_ms = self._core.capture_array()
        except Exception:
            return (None, None)

        if frame is None:
            return (None, ts_ms if ts_ms is not None else None)

        # Convert to OpenCV-friendly output for UI by default
        # - main stream often is already BGR888
        # - raw stream needs demosaic/scale-down
        try:
            frame = self._core.convert(frame, to="BGR888")
        except Exception:
            pass

        try:
            frame = self._core.postprocess(frame)
        except Exception:
            pass

        return frame, ts_ms

    def _allocate_framebuffer(self, frame=None) -> None:
        """Allocate (or re-allocate) the FrameBuffer.

        If `frame` is provided, allocate using its actual shape/dtype.
        Otherwise, allocate using best-effort expected output shape (BGR uint8).
        """

        # Prefer allocating from the actual produced frame.
        if frame is not None:
            try:
                shape = tuple(int(x) for x in frame.shape)
                dtype = frame.dtype
                self.buffer = FrameBuffer(
                    self._buffer_capacity,
                    shape,
                    dtype=dtype,
                    overwrite=self._buffer_overwrite,
                )
                self.capture = self.buffer
                return
            except Exception:
                # Fall back to expected allocation.
                pass

        try:
            out_res = self._core._configs.get("output_res", (-1, -1))
            out_w, out_h = int(out_res[0]), int(out_res[1])
        except Exception:
            out_w, out_h = (-1, -1)

        # Wrapper delivers OpenCV-friendly BGR by default.
        if out_w > 0 and out_h > 0:
            expected_shape = (out_h, out_w, 3)
        else:
            expected_shape = (
                int(getattr(self._core, "height", 480)),
                int(getattr(self._core, "width", 640)),
                3,
            )

        try:
            self.buffer = FrameBuffer(
                self._buffer_capacity,
                expected_shape,
                dtype="uint8",
                overwrite=self._buffer_overwrite,
            )
            self.capture = self.buffer
        except Exception:
            self.buffer = None
            self.capture = None

    def _camera_loop(self) -> None:

        exc_msg = ""
        try:
            ok = bool(self._core.open_cam())
        except Exception as exc:
            ok = False
            exc_msg = str(exc)

        if ok and self._core.cam_open:
            self._open_finished_evt.set()
        else:
            # Camera failed to open; reflect a stopped state.
            self._capture_evt.clear()
            self._reconfigure_evt.clear()
            self._loop_stop_evt.set()
            self._open_finished_evt.set()
            try:
                msg = "PiCam2:Failed to open camera" if not exc_msg else f"PiCam2:Failed to open camera: {exc_msg}"
                self.log.put_nowait((logging.CRITICAL, msg))
            except queue.Full:
                pass
            return

        # Allocate buffer immediately on open so consumers can poll .buffer.
        self._allocate_framebuffer()
        fb = self.buffer

        last_fps_t = time.perf_counter()
        num_frames = 0
        last_drop_warn_t = 0.0

        # Track what we've told the core about capture state (update-on-change).
        core_capturing_state = False

        # Local bindings (hot loop)
        capture_evt     = self._capture_evt
        reconfigure_evt = self._reconfigure_evt
        loop_stop_evt   = self._loop_stop_evt
        core            = self._core
        logq            = self.log
        drain_capture = self._drain_capture
        allocate_framebuffer = self._allocate_framebuffer

        # Start control thread
        if not (self._throttled_thread and self._throttled_thread.is_alive()):
            self._throttled_thread = threading.Thread(target=self._control_loop, daemon=True)
            self._throttled_thread.start()

        try:
            while not loop_stop_evt.is_set():

                # Apply reconfigure request (auto-stop capture -> apply -> auto-restart)
                # ----------------------------------------------------------------------
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

                        # If restart failed inside core, stop loop.
                        if not bool(getattr(self._core, "cam_open", False)):
                            raise RuntimeError("camera not open after reconfigure")

                        # Reset/reallocate buffer after configuration change.
                        try:
                            if self.buffer is not None:
                                self.buffer.clear()
                        except Exception:
                            pass
                        allocate_framebuffer()
                        fb = self.buffer

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

                # Capture and push frames into buffer
                # -----------------------------------

                # Update core capturing state only when it changes.
                capturing_now = bool(capture_evt.is_set())
                if capturing_now != core_capturing_state:
                    core_capturing_state = capturing_now
                    try:
                        core.capturing = capturing_now
                    except Exception:
                        pass

                    if capturing_now:
                        # reset fps window on capture start
                        last_fps_t = time.perf_counter()
                        num_frames = 0

                # Capture only when enabled
                if capturing_now:
                    now = time.perf_counter()
                    # drain captured frames
                    frame, ts_ms = drain_capture()
                    if frame is None:
                        time.sleep(0.001)
                        continue

                    num_frames += 1
                    self.frame_time = float(ts_ms if ts_ms is not None else (now * 1000.0))                    
                    
                    # Push into FrameBuffer without blocking
                    # --------------------------------------
                    try:
                        ok_push = fb.push(frame, self.frame_time)
                        if (not ok_push):
                            if (now - last_drop_warn_t) >= 1.0:
                                last_drop_warn_t = now
                                try:
                                    logq.put_nowait((logging.WARNING, "PiCam2:FrameBuffer is full; dropping frames"))
                                except queue.Full:
                                    pass

                    except Exception as exc1:
                        # Most likely a shape/dtype mismatch due to a stream/reconfigure change.
                        try:
                            logq.put_nowait((logging.WARNING, f"PiCam2:FrameBuffer push failed ({exc1}); reallocating"))
                        except queue.Full:
                            pass

                        # Reallocate to match this frame and retry once.
                        allocate_framebuffer(frame)
                        fb = self.buffer
                        try:
                            if fb is None:
                                raise RuntimeError("FrameBuffer reallocation failed")
                            ok_push = bool(fb.push(frame, self.frame_time))
                        except Exception as exc2:
                            try:
                                logq.put_nowait((logging.CRITICAL, f"PiCam2:FrameBuffer retry failed ({exc2}); stopping"))
                            except queue.Full:
                                pass
                            loop_stop_evt.set()
                            break

                    # Measure performance fps every 5seconds
                    # --------------------------------------

                    if (now - last_fps_t) >= 5.0:
                        self.measured_fps = num_frames / max((now - last_fps_t), 1e-6)
                        try:
                            logq.put_nowait((logging.INFO, f"PiCam2:FPS:{self.measured_fps}"))
                        except queue.Full:
                            pass
                        last_fps_t = now
                        num_frames = 0

                else:
                    self.measured_fps = 0.0
                    time.sleep(0.01)

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

    def _control_loop(self) -> None:
        """Separate thread for applying controls."""
        while not self._loop_stop_evt.is_set():
            time.sleep(0.1)  # 10 Hz control updates
            self._drain_controls()

    @property
    def measured_fps(self) -> float:
        # This Qt class emits fps via stats(); keep a convenience getter.
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
