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
# - pull(copy: bool|None): convenience wrapper around buffer.pull(); prefer
#   direct polling on `buffer`.
# - convertQimage(frame: np.ndarray) -> QImage | None   (expects OpenCV BGR order)
#
# Frame delivery model (NO queue semantics, NO frame signal):
# - Producers push frames into `buffer` without blocking.
# - Consumers poll (typically via a QTimer in the GUI thread):
#       if camera.buffer and camera.buffer.avail() > 0:
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
###############################################################################

###############################################################################
# Imports
###############################################################################

from __future__ import annotations
import time
import logging
import threading
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .picamera2core import PiCamera2Core, FrameBuffer

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
    """

    stats      = pyqtSignal(float)         # measured fps
    log        = pyqtSignal(int, str)      # logging level, message
    opened     = pyqtSignal()              # camera opened/configured
    started    = pyqtSignal()              # capture started
    stopped    = pyqtSignal()              # capture stopped

    def __init__(self, 
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

        self.buffer: FrameBuffer | None = None
        self.capture: FrameBuffer | None = None  # alias of self.buffer

        # Note: keep the core as the source of truth for stream sizes.

        # Camera loop lifecycle (thread runs even when not capturing)
        self._thread: threading.Thread | None = None
        self._loop_stop_evt = threading.Event()
        self._capture_evt = threading.Event()
        self._reconfigure_evt = threading.Event()
        self._opened_evt = threading.Event()  # signals camera opened attempt finished

        # Pending reconfigure-only changes (applied by loop)
        self._pending_camera_res: tuple[int, int] | None = None
        self._pending_flip: int | None = None

        # Runtime stats
        self.frame_time: float = 0.0  # last frame timestamp in ms (float)
        self._measured_fps: float = 0.0

        # Control requests from UI -> capture thread
        self._ctrl_q: "queue.SimpleQueue[dict]" = queue.SimpleQueue()

        # Core exists for the lifetime of this wrapper. The actual camera is
        # opened/configured inside the capture thread.
        self._core_log_q: "queue.Queue[tuple[int, str]]" = queue.Queue(maxsize=32)
        self._core = PiCamera2Core(
            self._configs,
            camera_num=self._camera_num,
            res=self._res_override,
            exposure=self._exposure_override,
            log_queue=self._core_log_q,
        )

        # Latest-frame fields removed: consumers poll FrameBuffer instead.


    # ------------------------------------------------------------------
    # Lifecycle (unified with Qt wrapper semantics)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def start(self):
        """Enable capturing (camera loop remains running)."""
        # Ensure camera loop is running
        if not (self._thread and self._thread.is_alive()):
            try:
                self.open_cam()
            except Exception as e:
                logging.error(f"PiCam2: Failed to open camera: {e}")

        if not self.cam_open:
            try:
                self.log.emit(logging.CRITICAL, "PiCam2: Cannot start capture; camera not open")
            except Exception:
                pass
            return

        self._capture_evt.set()


    @pyqtSlot()
    def stop(self):
        """Disable capturing but keep the camera open and loop running."""
        self._capture_evt.clear()

    def join(self, timeout: float | None = None):
        """Block until the capture thread exits (non-Qt helper)."""
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
        capture.
        """
        if self._thread and self._thread.is_alive():
            return bool(self.cam_open)

        self._opened_evt.clear()
        self._loop_stop_evt.clear()
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._thread.start()

        if timeout is not None:
            try:
                self._opened_evt.wait(timeout=float(timeout))
            except Exception:
                pass

        return bool(self.cam_open)

    def close_cam(self, timeout: float | None = 2.0):
        """Disable capturing, stop the loop thread, and close the camera."""
        self.stop()
        self._loop_stop_evt.set()
        self.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Camera control helpers (Qt/non-Qt parity)
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def set_exposure_us(self, exposure_us: int):
        # Don’t call picam2 here; just enqueue a request
        self._ctrl_q.put({"AeEnable": False, "ExposureTime": int(exposure_us)})

    @pyqtSlot(bool)
    def set_auto_exposure(self, enabled: bool):
        self._ctrl_q.put({"AeEnable": bool(enabled)})

    @pyqtSlot(int)
    @pyqtSlot(str)
    def set_aemeteringmode(self, mode):
        """Set AE metering mode (int or friendly string)."""
        try:
            meter_val = self._core._parse_aemeteringmode(mode)
            self._ctrl_q.put({"AeMeteringMode": int(meter_val)})
        except Exception:
            pass

    @pyqtSlot(int)
    @pyqtSlot(str)
    def set_awbmode(self, mode):
        """Set AWB mode (int or friendly string)."""
        try:
            awb_val = self._core._parse_awbmode(mode)
            self._ctrl_q.put({"AwbMode": int(awb_val)})
        except Exception:
            pass

    @pyqtSlot(bool)
    def set_auto_wb(self, enabled: bool):
        self._ctrl_q.put({"AwbEnable": bool(enabled)})

    @pyqtSlot(float)
    def set_framerate(self, fps: float):
        self._ctrl_q.put({"FrameRate": float(fps)})

    @pyqtSlot(int)
    def set_flip(self, flip: int):
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
                self.log.emit(logging.INFO, f"PiCam2:flip set to {f}; will reconfigure")
            except Exception:
                pass
        except Exception:
            pass

    def set_resolution(self, res: tuple[int, int]):
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
            self.log.emit(logging.INFO, f"PiCam2:resolution set to {w}x{h}; will reconfigure")
        except Exception:
            return

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
        finally:
            self._drain_core_logs()

    def get_control(self, name: str):
        try:
            return self._core.get_control(name)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal camera loop
    # ------------------------------------------------------------------

    def _drain_core_logs(self) -> None:
        """Forward any queued core logs to the Qt `log` signal."""
        q = self._core_log_q
        while True:
            try:
                level, msg = q.get_nowait()
            except Exception:
                break
            try:
                self.log.emit(int(level), str(msg))
            except Exception:
                pass

    def _drain_controls(self):
        """Apply any pending control requests (last-write-wins)."""
        merged: dict = {}
        while True:
            try:
                d = self._ctrl_q.get_nowait()
            except Exception:
                break
            if isinstance(d, dict):
                merged.update(d)

        if not merged:
            return

        try:
            self._core.set_controls(merged)
        except Exception:
            pass

    def _drain_capture(self) -> tuple[np.ndarray | None, float | None]:
        """Capture and process a single frame (if available)."""
        try:
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
            try:
                self.opened.emit()
            except Exception:
                pass
            self._opened_evt.set()
        else:
            # Camera failed to open; reflect a stopped state.
            self._capture_evt.clear()
            self._opened_evt.set()
            try:
                self.stopped.emit()
            except Exception:
                pass
            msg = "PiCam2:Failed to open camera" if not exc_msg else f"PiCam2:Failed to open camera: {exc_msg}"
            try:
                self.log.emit(logging.CRITICAL, msg)
            except Exception:
                pass
            return

        self._drain_core_logs()

        # Allocate buffer immediately on open so consumers can poll .buffer.
        self._allocate_framebuffer()

        last_fps_t = time.perf_counter()
        num_frames = 0

        # Track capture state transitions for started/stopped signals
        capturing_prev = False

        # Track what we've told the core about capture state (update-on-change).
        core_capturing_state = False

        # Local bindings (hot loop) - mirror non-Qt wrapper structure
        loop_stop_evt = self._loop_stop_evt
        capture_evt = self._capture_evt
        reconfigure_evt = self._reconfigure_evt
        core = self._core
        drain_controls = self._drain_controls
        drain_core_logs = self._drain_core_logs
        drain_capture = self._drain_capture

        try:
            while not loop_stop_evt.is_set():
                # apply pending controls as often as possible
                drain_controls()

                # forward any core logs
                drain_core_logs()

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
                        self._allocate_framebuffer()

                        self.opened.emit()
                        drain_core_logs()

                    except Exception as exc:
                        try:
                            self.log.emit(logging.CRITICAL, f"PiCam2:Reconfigure failed: {exc}")
                        except Exception:
                            pass
                        loop_stop_evt.set()
                        break

                    if was_capturing:
                        capture_evt.set()
                    continue

                # Capture and push frames into buffer
                # -----------------------------------

                # Update core capturing state only when it changes.
                capturing_now = bool(capture_evt.is_set())

                if capturing_now != capturing_prev:
                    capturing_prev = capturing_now

                    # Update core capture state on transitions.
                    core_capturing_state = capturing_now
                    try:
                        core.capturing = capturing_now
                    except Exception:
                        pass
                    if capturing_now:
                        # reset fps window on capture start
                        last_fps_t = time.perf_counter()
                        num_frames = 0
                        self.started.emit()
                    else:
                        self.stopped.emit()

                if capturing_now:
                    now = time.perf_counter()
                    # drain captured frames
                    frame, ts_ms = drain_capture()
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    num_frames += 1
                    if ts_ms is None:
                        ts_ms = float(now * 1000.0)
                    else:
                        ts_ms = float(ts_ms)

                    self.frame_time = ts_ms

                    # Push into FrameBuffer without blocking.
                    fb = self.buffer
                    if fb is None:
                        self._allocate_framebuffer(frame)
                        fb = self.buffer
                        if fb is None:
                            try:
                                self.log.emit(logging.CRITICAL, "PiCam2:No FrameBuffer available; stopping")
                            except Exception:
                                pass
                            loop_stop_evt.set()
                            break

                    try:
                        ok_push = bool(fb.push(frame, ts_ms))
                        if (not ok_push) and (not self._buffer_overwrite):
                            try:
                                self.log.emit(logging.WARNING, "PiCam2:FrameBuffer is full; dropping frame")
                            except Exception:
                                pass
                    except Exception as exc1:
                        # Most likely a shape/dtype mismatch due to reconfigure.
                        try:
                            self.log.emit(logging.WARNING, f"PiCam2:FrameBuffer push failed ({exc1}); reallocating")
                        except Exception:
                            pass

                        self._allocate_framebuffer(frame)
                        fb = self.buffer
                        try:
                            if fb is None:
                                raise RuntimeError("FrameBuffer reallocation failed")
                            ok_push = bool(fb.push(frame, ts_ms))
                        except Exception as exc2:
                            try:
                                self.log.emit(logging.CRITICAL, f"PiCam2:FrameBuffer retry failed ({exc2}); stopping")
                            except Exception:
                                pass
                                loop_stop_evt.set()
                            break

                    # low_latency controls emission semantics:
                    # - low_latency=True typically uses a size-1 buffer so consumers
                    #   always see the most recent frame when polling.

                    # Stats every ~5s
                    if (now - last_fps_t) >= 5.0:
                        self._measured_fps = num_frames / (now - last_fps_t)
                        self.stats.emit(self._measured_fps)
                        num_frames = 0
                        last_fps_t = now
                else:
                    # Idle: keep loop responsive without busy-spinning
                    time.sleep(0.01)

        finally:
            self._core.capturing = False
            self._capture_evt.clear()
            self._core.close_cam()
            self._drain_core_logs()
            # If we exit while capture was enabled, emit a final stopped.
            if capturing_prev:
                try:
                    self.stopped.emit()
                except Exception:
                    pass

    def pull(self, copy: bool | None = None):
        """Convenience pull from buffer.

        Prefer direct buffer usage:
            if camera.buffer and camera.buffer.avail() > 0:
                frame, ts_ms = camera.buffer.pull(copy=False)
        """
        fb = self.buffer
        if fb is None:
            return (None, None)
        if copy is None:
            copy = bool(self._buffer_copy_on_pull)
        return fb.pull(copy=bool(copy))

    def __getattr__(self, name: str):
        """Convenience: delegate unknown attributes to the core."""
        return getattr(self._core, name)

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

    def convertQimage(self, frame) -> QImage | None:
        """Convert an OpenCV-style numpy frame to QImage for display.

        Expected input: uint8 HxWx3 in BGR (OpenCV order) or a 2D grayscale image.
        Returns a deep-copied QImage (safe after returning).
        """
        try:
            if frame is None:
                return None
            if not isinstance(frame, np.ndarray):
                return None

            if frame.ndim == 2:
                h, w = frame.shape
                return QImage(
                    frame.data,
                    w,
                    h,
                    int(frame.strides[0]),
                    _QIMAGE_FMT_GRAY8,
                ).copy()

            if frame.ndim != 3 or frame.shape[2] < 3:
                return None

            # Assume BGR input and swap to RGB for Qt
            bgr = frame[:, :, :3]
            rgb = bgr[:, :, ::-1]
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
