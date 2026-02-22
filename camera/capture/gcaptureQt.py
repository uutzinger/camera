###############################################################################
# GStreamer video capture (Qt wrapper)
#
# Threaded capture wrapper around gCapture for Qt5/Qt6.
#
# Urs Utzinger
# GPT-5.2
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: gCaptureQt(QObject)
#
# Threaded capture wrapper around `gCapture` for Qt.
#
# Public attributes:
# - capture: Queue[(ts_ms: float, frame: np.ndarray)]
# - log_queue: Queue[(level: int, message: str)]
#
# Signals (Qt):
# - stats(measured_fps: float)
# - log(level: int, message: str)
# - opened()
# - started()
# - stopped()
#
# Public methods:
# - start() / stop(): enable/disable capturing
# - close(timeout: float | None = 2.0) -> None
# - join(timeout: float | None = None) -> None
# - convertQimage(frame: np.ndarray) -> QImage | None
#
# Supported Config Parameters (configs dict)
# ------------------------------------------
# Same as gcapture.py / gCapture_Config.md.
###############################################################################

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .gcapture import gCapture

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


class gCaptureQt(QObject):
    """Qt wrapper around gCapture."""

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
        self._res = res
        self._exposure = exposure
        self._queue_size = int(queue_size)

        self._cam_lock = threading.Lock()
        self._camera: gCapture | None = None
        self._camera_started_once = False

        self.capture: Queue | None = None
        self.log_queue: Queue | None = None

        self._relay_stop_evt = threading.Event()
        self._relay_thread = threading.Thread(target=self._relay_loop, daemon=True)

        self._create_camera()
        self._relay_thread.start()

    def _create_camera(self) -> None:
        cam = gCapture(
            self._configs,
            camera_num=self._camera_num,
            res=self._res,
            exposure=self._exposure,
            queue_size=self._queue_size,
        )
        self._camera = cam
        self.capture = cam.capture
        self.log_queue = cam.log

    @pyqtSlot()
    def start(self) -> bool:
        with self._cam_lock:
            cam = self._camera
            if cam is None:
                self._create_camera()
                cam = self._camera

            # gCapture subclasses Thread, so restart requires a new instance.
            if cam is not None and self._camera_started_once and (not cam.is_alive()):
                try:
                    cam.close_cam()
                except Exception:
                    pass
                self._create_camera()
                cam = self._camera
                self._camera_started_once = False

            if cam is None:
                return False

            if not getattr(cam, "cam_open", False):
                try:
                    cam.open_cam()
                except Exception:
                    pass
                self.capture = cam.capture
                self.log_queue = cam.log

            if not getattr(cam, "cam_open", False):
                try:
                    self.log.emit(logging.ERROR, "gCaptureQt:Camera not open; cannot start")
                except Exception:
                    pass
                return False

            try:
                cam.start()
                self._camera_started_once = True
            except RuntimeError:
                # Defensive path for Python thread restart errors.
                self._create_camera()
                cam = self._camera
                if cam is None or not getattr(cam, "cam_open", False):
                    return False
                cam.start()
                self._camera_started_once = True

        try:
            self.opened.emit()
            self.started.emit()
        except Exception:
            pass
        return True

    @pyqtSlot()
    def stop(self, timeout: float | None = 2.0) -> None:
        with self._cam_lock:
            cam = self._camera
        if cam is not None and cam.is_alive():
            cam.stop()
            try:
                cam.join(timeout=timeout)
            except Exception:
                pass
        try:
            self.stopped.emit()
        except Exception:
            pass

    def join(self, timeout: float | None = None) -> None:
        with self._cam_lock:
            cam = self._camera
        if cam is not None and cam.is_alive():
            cam.join(timeout=timeout)

    @pyqtSlot()
    def close(self, timeout: float | None = 2.0) -> None:
        self.stop(timeout=timeout)

        with self._cam_lock:
            cam = self._camera
            self._camera = None

        if cam is not None:
            try:
                cam.close_cam()
            except Exception:
                pass

        self._relay_stop_evt.set()
        if self._relay_thread.is_alive():
            self._relay_thread.join(timeout=timeout)

    def _relay_loop(self) -> None:
        last_stats_t = time.perf_counter()
        while not self._relay_stop_evt.is_set():
            with self._cam_lock:
                cam = self._camera

            if cam is None:
                time.sleep(0.05)
                continue

            try:
                level, msg = cam.log.get(timeout=0.2)
                try:
                    self.log.emit(int(level), str(msg))
                except Exception:
                    pass
            except Empty:
                pass
            except Exception:
                pass

            now = time.perf_counter()
            if (now - last_stats_t) >= 5.0:
                try:
                    self.stats.emit(float(cam.measured_fps))
                except Exception:
                    pass
                last_stats_t = now

    @property
    def cam_open(self) -> bool:
        with self._cam_lock:
            cam = self._camera
        return bool(cam is not None and getattr(cam, "cam_open", False))

    @property
    def measured_fps(self) -> float:
        with self._cam_lock:
            cam = self._camera
        if cam is None:
            return 0.0
        try:
            return float(cam.measured_fps)
        except Exception:
            return 0.0

    def convertQimage(self, frame: np.ndarray):
        """Convert a BGR numpy image to QImage."""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        if frame.ndim == 2:
            bytes_per_line = w
            return QImage(frame.data, w, h, bytes_per_line, _QIMAGE_FMT_GRAY8).copy()

        if frame.ndim == 3 and frame.shape[2] >= 3:
            bytes_per_line = int(frame.shape[1] * frame.shape[2])
            rgb = frame[:, :, ::-1]
            return QImage(rgb.data, w, h, bytes_per_line, _QIMAGE_FMT_RGB888).copy()

        return None

    def __getattr__(self, name: str):
        with self._cam_lock:
            cam = self._camera
        if cam is None:
            raise AttributeError(name)
        return getattr(cam, name)
