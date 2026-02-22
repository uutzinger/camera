###############################################################################
# RTSP video capture (threaded wrapper)
#
# Threaded capture wrapper around RtspCore.
#
# Urs Utzinger
# GPT-5.2
#
# 2026 First Release
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: rtspCapture
#
# Threaded capture wrapper around `RtspCore`.
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
# - close(timeout: float | None = 2.0): stop camera loop thread and close stream
# - log_stream_options(): delegated to core
# - set_output_res(res: tuple[int,int])
# - set_resolution(res: tuple[int,int])  # alias of set_output_res
# - set_flip(flip: int)
# - get_control(name: str) -> Any
#
# - __getattr__: delegates unknown attributes/properties to the core.
#
# Frame delivery model (NO queue semantics):
# - Producers push frames into `buffer` without blocking.
# - Consumers poll:
#       if camera.buffer and camera.buffer.avail > 0:
#           frame, ts_ms = camera.buffer.pull(copy=True)
#
# Convenience properties:
# - cam_open: bool
# - measured_fps: float
#
# Supported Config Parameters (configs dict)
# ------------------------------------------
# See rtspcore.py for full list.
#
# Note:
# - Stream is opened in __init__ with the provided configs. start()/stop() only
#   control frame capture, not stream open/close.
###############################################################################

from __future__ import annotations

import threading
from queue import Queue
import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .rtspcore import RtspCore
from .framebuffer import FrameBuffer


class rtspCapture:
    """
    Threaded capture wrapper around RtspCore.

    Public properties:
    - cam_open: bool
        Indicates if the stream is open.
    - measured_fps: float
        Returns the measured frames per second.
    """

    def __init__(
        self,
        configs: dict | None,
        rtsp: str | None = None,
        gpu: bool = False,
        queue_size: int = 32,
    ) -> None:
        self._configs = configs or {}
        self._rtsp_override = rtsp
        self._gpu_override = gpu

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

        self.log = Queue(maxsize=32)

        # Runtime stats
        self._measured_fps: float = 0.0

        self._core = RtspCore(
            self._configs,
            rtsp=self._rtsp_override,
            gpu=self._gpu_override,
            log_queue=self.log,
        )

        # Camera loop lifecycle
        self._capture_thread: threading.Thread | None = None
        self._loop_stop_evt = threading.Event()
        self._capture_evt = threading.Event()
        self._open_finished_evt = threading.Event()

        # Open the stream immediately
        self._core.open_cam()
        self.buffer = self._core.buffer
        self.capture = self.buffer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the capture loop (does not reopen stream)."""
        if not self._core.cam_open:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, "RTSP:Stream not open; cannot start capture"))
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
        """Stop the capture loop (does not close stream)."""
        self._capture_evt.clear()
        self._loop_stop_evt.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        self._capture_thread = None

    def join(self, timeout: float | None = None):
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)

    def close(self, timeout: float | None = 2.0):
        """Stop capture loop and close stream."""
        self._capture_evt.clear()
        self._loop_stop_evt.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
        self._capture_thread = None
        self._core.close_cam()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def log_stream_options(self) -> None:
        self._core.log_stream_options()

    def set_output_res(self, res: tuple[int, int]) -> None:
        try:
            self._core.set_output_res(res)
        except Exception as exc:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"RTSP:set_output_res failed: {exc}"))
            except Exception:
                pass

    def set_resolution(self, res: tuple[int, int]) -> None:
        """Alias of set_output_res for API parity."""
        self.set_output_res(res)

    def set_flip(self, flip: int) -> None:
        try:
            self._core.set_flip(flip)
        except Exception as exc:
            try:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"RTSP:set_flip failed: {exc}"))
            except Exception:
                pass

    def get_control(self, name: str):
        return self._core.get_control(name)

    # ------------------------------------------------------------------
    # Internal camera loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        loop_stop_evt = self._loop_stop_evt
        capture_evt = self._capture_evt
        core = self._core
        logq = self.log

        last_drop_warn_t = 0.0
        fps_measurement_interval = 5.0
        fps_frame_count = 0
        fps_start_time = time.perf_counter()

        fb = core.buffer

        try:
            while not loop_stop_evt.is_set():
                if capture_evt.is_set():
                    loop_start = time.perf_counter()
                    try:
                        frame, ts_ms = core.capture_array()
                    except Exception as exc:
                        try:
                            logq.put_nowait((logging.WARNING, f"RTSP:capture_array failed: {exc}"))
                        except Exception:
                            pass
                        time.sleep(0.001)
                        continue

                    if frame is None:
                        time.sleep(0.001)
                        continue

                    frame_time = float(ts_ms if ts_ms is not None else (loop_start * 1000.0))

                    if fb is None:
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
                        except Exception as exc:
                            try:
                                logq.put_nowait((logging.CRITICAL, f"RTSP:FrameBuffer alloc failed ({exc}); stopping"))
                            except Exception:
                                pass
                            loop_stop_evt.set()
                            break

                    try:
                        ok_push = bool(fb.push(frame, frame_time)) if fb is not None else False
                        if not ok_push:
                            now = time.perf_counter()
                            if (now - last_drop_warn_t) >= 1.0:
                                last_drop_warn_t = now
                                try:
                                    logq.put_nowait((logging.WARNING, "RTSP:FrameBuffer is full; dropping frames"))
                                except RuntimeError:
                                    pass
                        else:
                            fps_frame_count += 1

                    except Exception as exc1:
                        try:
                            logq.put_nowait((logging.WARNING, f"RTSP:FrameBuffer push failed ({exc1}); reallocating"))
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
                                logq.put_nowait((logging.CRITICAL, f"RTSP:FrameBuffer retry failed ({exc2}); stopping"))
                            except Exception:
                                pass
                            loop_stop_evt.set()
                            break

                    elapsed = loop_start - fps_start_time
                    if elapsed >= fps_measurement_interval:
                        self._measured_fps = fps_frame_count / elapsed
                        fps_start_time = loop_start
                        fps_frame_count = 0
                else:
                    self._measured_fps = 0.0
                    time.sleep(0.01)
        finally:
            self._core.close_cam()

    # ------------------------------------------------------------------
    # Convenience properties / delegation
    # ------------------------------------------------------------------

    @property
    def cam_open(self) -> bool:
        return bool(self._core.cam_open)

    @property
    def measured_fps(self) -> float:
        return float(self._measured_fps)

    def __getattr__(self, name: str):
        return getattr(self._core, name)
