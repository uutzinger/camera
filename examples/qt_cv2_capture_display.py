"""Qt5/Qt6 example: OpenCV cv2CaptureQt display with Start/Stop.

This is the Qt counterpart to examples/cv2_capture_display.py.

- Uses camera.capture.cv2captureQt.cv2CaptureQt
- Displays frames in a Qt window
- Provides a Start/Stop toggle button

Run:
    python3 examples/qt_cv2_capture_display.py
"""

import sys
import time
import logging

try:
    from PyQt6.QtCore import Qt, QTimer  # type: ignore
    from PyQt6.QtGui import QPixmap  # type: ignore
    from PyQt6.QtWidgets import (  # type: ignore
        QApplication,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QWidget,
    )
except Exception:  # pragma: no cover
    from PyQt5.QtCore import Qt, QTimer  # type: ignore
    from PyQt5.QtGui import QPixmap  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QApplication,
        QLabel,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QWidget,
    )


# Enum compatibility between Qt5 and Qt6
try:
    _ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    _KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    _FAST_TRANSFORM = Qt.TransformationMode.FastTransformation
except Exception:  # pragma: no cover
    _ALIGN_CENTER = Qt.AlignCenter
    _KEEP_ASPECT = Qt.KeepAspectRatio
    _FAST_TRANSFORM = Qt.FastTransformation


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("cv2Capture Qt Display")

        # Match cv2_capture_display defaults (can be edited)
        self.configs = {
            ################################################################
            # OpenCV capture configuration
            ################################################################
            "camera_res": (640, 480),
            "exposure": -1,            # backend-dependent units; -1 for auto
            "fps": 60,
            "autoexposure": -1,        # -1 leave unchanged, 0 manual, 1 auto
            "fourcc": "MJPG",          # MJPG reduces USB bandwidth on many webcams
            "buffersize": 4,
            "buffer_overwrite": True,
            "output_res": (-1, -1),    # (-1,-1): output == input
            "flip": 0,
            "displayfps": 30,
            "convert_rgb": True,       # backend returns BGR; set False to request raw
        }

        displayfps = float(self.configs.get("displayfps", 0) or 0)
        capture_fps = float(self.configs.get("fps", 0) or 0)
        if displayfps <= 0:
            self._display_interval = 0.0
        elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
            self._display_interval = 0.0
        else:
            self._display_interval = 1.0 / displayfps
        self._last_display = 0.0

        from camera.capture.cv2captureQt import cv2CaptureQt

        self.camera = cv2CaptureQt(self.configs, camera_num=0)
        self.camera.stats.connect(self.on_stats)
        self.camera.log.connect(self.on_log)
        self.camera.started.connect(self.on_started)
        self.camera.stopped.connect(self.on_stopped)

        # Poll/drain buffer in GUI thread (no frameReady signal)
        self._poll_timer = QTimer(self)
        interval_ms = 0 if self._display_interval <= 0.0 else max(1, int(self._display_interval * 1000.0))
        self._poll_timer.setInterval(interval_ms)
        self._poll_timer.timeout.connect(self.on_poll)

        # UI
        self.video = QLabel("No video")
        self.video.setAlignment(_ALIGN_CENTER)
        self.video.setMinimumSize(320, 240)

        self.btn_toggle = QPushButton("Start")
        self.btn_toggle.clicked.connect(self.start_stream)

        button_row = QHBoxLayout()
        button_row.addWidget(self.btn_toggle)
        button_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.video, stretch=1)
        layout.addLayout(button_row)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        # Logging
        self._logger = logging.getLogger("cv2_capture_qt_display")

        # FPS tracking
        self._capture_fps = 0.0
        self._display_fps = 0.0
        self._display_frames = 0
        self._display_fps_last_t = time.perf_counter()
        self._update_statusbar()

    def _update_statusbar(self) -> None:
        self.statusBar().showMessage(
            f"Capture FPS: {self._capture_fps:.1f} | Display FPS: {self._display_fps:.1f}"
        )

    def _set_toggle_to_start(self):
        self.btn_toggle.setText("Start")
        try:
            self.btn_toggle.clicked.disconnect(self.stop_stream)
        except Exception:
            pass
        try:
            self.btn_toggle.clicked.connect(self.start_stream)
        except Exception:
            pass

    def _set_toggle_to_stop(self):
        self.btn_toggle.setText("Stop")
        try:
            self.btn_toggle.clicked.disconnect(self.start_stream)
        except Exception:
            pass
        try:
            self.btn_toggle.clicked.connect(self.stop_stream)
        except Exception:
            pass

    def start_stream(self):
        self._set_toggle_to_stop()
        self.camera.start()

    def stop_stream(self):
        self.btn_toggle.setEnabled(False)
        self.camera.stop()

    def on_started(self):
        self._logger.info("Capture started")
        try:
            self.camera.log_stream_options()
        except Exception:
            pass
        try:
            self._poll_timer.start()
        except Exception:
            pass
        self.btn_toggle.setEnabled(True)

    def on_stopped(self):
        self._logger.info("Capture stopped")
        try:
            self._poll_timer.stop()
        except Exception:
            pass
        self._set_toggle_to_start()
        self.btn_toggle.setEnabled(True)

    def on_stats(self, fps: float):
        self._capture_fps = float(fps)
        self._update_statusbar()

    def on_log(self, level: int, message: str):
        try:
            self._logger.log(int(level), "%s", message)
        except Exception:
            pass

    def on_poll(self):
        now = time.perf_counter()
        if self._display_interval > 0.0 and (now - self._last_display) < self._display_interval:
            return

        if self.camera.buffer is None or self.camera.buffer.avail <= 0:
            return

        frame, _ts = self.camera.buffer.pull(copy=False)
        if frame is None:
            return

        qimg = self.camera.convertQimage(frame)
        if qimg is None:
            return

        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.video.size(), _KEEP_ASPECT, _FAST_TRANSFORM)
        self.video.setPixmap(pix)

        self._last_display = now
        self._display_frames += 1

        if (now - self._display_fps_last_t) >= 2.0:
            self._display_fps = self._display_frames / (now - self._display_fps_last_t)
            self._display_frames = 0
            self._display_fps_last_t = now
            self._update_statusbar()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
