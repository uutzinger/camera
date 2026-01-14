"""Qt5/Qt6 example: synthetic Picamera2 wrapper display with Start/Stop.

Synthetic counterpart to:
- examples/qt_picamera2_capture_display.py

Uses the same Qt wrapper + FrameBuffer polling, but enables synthetic frame
generation in PiCamera2Core via configs['test_pattern'].

Run:
    python3 examples/qt_picamera2_synthetic_capture_display.py
"""

import os
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


try:
    _ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    _KEEP_ASPECT = Qt.AspectRatioMode.KeepAspectRatio
    _FAST_TRANSFORM = Qt.TransformationMode.FastTransformation
except Exception:  # pragma: no cover
    _ALIGN_CENTER = Qt.AlignCenter
    _KEEP_ASPECT = Qt.KeepAspectRatio
    _FAST_TRANSFORM = Qt.FastTransformation


os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PiCamera2 Qt Display (synthetic)")

        self.configs = {
            "mode": "main",
            "camera_res": (640, 480),
            "exposure": 0,
            # Stress-test: high synthetic FPS. 0 = unpaced.
            "fps": 1000,
            "autoexposure": 1,
            "aemeteringmode": "center",
            "autowb": 1,
            "awbmode": "auto",
            "format": "BGR3",
            "stream_policy": "default",
            "low_latency": True,
            "buffersize": 4,
            "output_res": (-1, -1),
            "flip": 0,
            "displayfps": 30,
            "test_pattern": "gradient",
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

        from camera.capture.picamera2captureQt import piCamera2CaptureQt

        self.capture = piCamera2CaptureQt(self.configs, camera_num=0)
        self.capture.stats.connect(self.on_stats)
        self.capture.log.connect(self.on_log)
        self.capture.started.connect(self.on_started)
        self.capture.stopped.connect(self.on_stopped)

        self._poll_timer = QTimer(self)
        interval_ms = 0 if self._display_interval <= 0.0 else max(1, int(self._display_interval * 1000.0))
        self._poll_timer.setInterval(interval_ms)
        self._poll_timer.timeout.connect(self.on_poll)

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

        self._logger = logging.getLogger("picamera2_synth_qt_display")

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
        self.capture.start()

    def stop_stream(self):
        self.btn_toggle.setEnabled(False)
        self.capture.stop()

    def on_started(self):
        self._logger.info("Capture started")
        try:
            self.capture.log_stream_options()
        except Exception:
            pass
        try:
            self._poll_timer.start()
        except Exception:
            pass
        self.btn_toggle.setEnabled(True)
        self._set_toggle_to_stop()

    def on_stopped(self):
        self._logger.info("Capture stopped")
        try:
            self._poll_timer.stop()
        except Exception:
            pass
        self.btn_toggle.setEnabled(True)
        self._set_toggle_to_start()

    def on_log(self, level: int, message: str):
        try:
            self._logger.log(int(level), str(message))
        except Exception:
            self._logger.info("%s", message)

    def on_stats(self, fps: float):
        try:
            self._capture_fps = float(fps)
        except Exception:
            self._capture_fps = 0.0
        self._update_statusbar()

    def on_poll(self):
        buf = getattr(self.capture, "buffer", None)
        if buf is None:
            return

        last = None
        try:
            while buf.avail() > 0:
                last = buf.pull(copy=False)
        except Exception:
            last = None

        if not last:
            return

        frame, _ts_ms = last
        try:
            frame = frame.copy()
        except Exception:
            pass

        try:
            img = self.capture.convertQimage(frame)
            if img is None:
                raise ValueError("convertQimage() returned None")
            pix = QPixmap.fromImage(img)
            self.video.setPixmap(pix.scaled(self.video.size(), _KEEP_ASPECT, _FAST_TRANSFORM))

            now = time.perf_counter()
            self._display_frames += 1
            if (now - self._display_fps_last_t) >= 2.0:
                dt = now - self._display_fps_last_t
                self._display_fps = self._display_frames / max(dt, 1e-6)
                self._display_frames = 0
                self._display_fps_last_t = now
                self._update_statusbar()

        except Exception as exc:
            self._logger.debug("Display update failed: %s", exc)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(960, 540)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
