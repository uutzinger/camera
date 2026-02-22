"""Qt5/Qt6 example: Basler pylonCaptureQt display with Start/Stop.

This mirrors examples/qt_cv2_capture_display.py but uses the pypylon backend.
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

        self.setWindowTitle("Basler Pylon Qt Display")

        from configs.pylon_configs import configs

        self.configs = dict(configs)
        self.configs.setdefault("displayfps", 30)
        self.configs.setdefault("convert_format", "Mono8")

        displayfps = float(self.configs.get("displayfps", 0) or 0)
        capture_fps = float(self.configs.get("fps", 0) or 0)
        if displayfps <= 0:
            self._display_interval = 0.0
        elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
            self._display_interval = 0.0
        else:
            self._display_interval = 1.0 / displayfps
        self._last_display = 0.0

        from camera.capture.pyloncaptureQt import pylonCaptureQt

        self.camera = pylonCaptureQt(self.configs, camera_num=0)
        self.camera.stats.connect(self.on_stats)
        self.camera.log.connect(self.on_log)
        self.camera.started.connect(self.on_started)
        self.camera.stopped.connect(self.on_stopped)

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

        self._last_stats_t = time.perf_counter()
        self._measured_fps = 0.0

    def start_stream(self):
        if self.btn_toggle.text().lower().startswith("start"):
            ok = self.camera.start()
            if ok:
                self._poll_timer.start()
        else:
            self._poll_timer.stop()
            self.camera.stop()

    def on_started(self):
        self.btn_toggle.setText("Stop")

    def on_stopped(self):
        self.btn_toggle.setText("Start")

    def on_stats(self, fps: float):
        self._measured_fps = float(fps)

    def on_log(self, level: int, msg: str):
        logging.getLogger("qt_pylon").log(level, "%s", msg)

    def on_poll(self):
        now = time.perf_counter()
        if self._display_interval > 0.0 and (now - self._last_display) < self._display_interval:
            return

        if not self.camera.buffer or self.camera.buffer.avail <= 0:
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

        if now - self._last_stats_t >= 5.0:
            logging.getLogger("qt_pylon").info("Capture FPS: %.1f", self._measured_fps)
            self._last_stats_t = now

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        try:
            self._poll_timer.stop()
            self.camera.stop()
            self.camera.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(960, 640)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
