"""Qt5/Qt6 example: RTP display with Start/Stop.

This is the Qt counterpart to examples/rtp_display.py.

- Uses camera.capture.rtpcaptureQt.rtpCaptureQt
- Displays frames in a Qt window
- Provides a Start/Stop toggle button

Run:
    python3 examples/qt_rtp_capture_display.py

Notes:
- If GI/GStreamer is available, it will be used.
- Otherwise FFmpeg fallback requires an SDP file (rtp_sdp) and a frame size.
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

        self.setWindowTitle('RTP Qt Display')

        # Match rtp_display defaults (can be edited)
        self.configs = {
            ################################################################
            # RTP capture configuration
            ################################################################
            'port'           : 554,
            'prefer_gi'      : True,
            # FFmpeg fallback requires SDP + frame size
            'rtp_sdp'        : 'examples/rtp_h264_pt96.sdp',
            'camera_res'     : (640, 480),
            'output_res'     : (-1, -1),       # Output resolution
            'flip'           : 0,              # 0=norotation
            'displayfps'     : 30              # consumer-side display throttle
        }

        displayfps = float(self.configs.get('displayfps', 0) or 0)
        capture_fps = float(self.configs.get('fps', 0) or 0)
        if displayfps <= 0:
            self._display_interval = 0.0
        elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
            self._display_interval = 0.0
        else:
            self._display_interval = 1.0 / displayfps
        self._last_display = 0.0

        from camera.capture.rtpcaptureQt import rtpCaptureQt

        self.camera = rtpCaptureQt(self.configs, port=self.configs.get('port'))
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
        self.video = QLabel('No video')
        self.video.setAlignment(_ALIGN_CENTER)
        self.video.setMinimumSize(320, 240)

        self.btn_toggle = QPushButton('Start')
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
        self._logger = logging.getLogger('rtp_capture_qt_display')

        # FPS tracking
        self._capture_fps = 0.0
        self._display_fps = 0.0
        self._display_frames = 0
        self._display_fps_last_t = time.perf_counter()
        self._update_statusbar()

    def _update_statusbar(self) -> None:
        self.statusBar().showMessage(
            f'Capture FPS: {self._capture_fps:.1f} | Display FPS: {self._display_fps:.1f}'
        )

    def _set_toggle_to_start(self):
        self.btn_toggle.setText('Start')
        try:
            self.btn_toggle.clicked.disconnect(self.stop_stream)
        except Exception:
            pass
        try:
            self.btn_toggle.clicked.connect(self.start_stream)
        except Exception:
            pass

    def _set_toggle_to_stop(self):
        self.btn_toggle.setText('Stop')
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
        self._logger.info('Capture started')
        try:
            self.camera.log_stream_options()
        except Exception:
            pass
        try:
            self._poll_timer.start()
        except Exception:
            pass
        self.btn_toggle.setEnabled(True)
        self._set_toggle_to_stop()

    def on_stopped(self):
        self._logger.info('Capture stopped')
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
            self._logger.info('%s', message)

    def on_stats(self, fps: float):
        try:
            self._capture_fps = float(fps)
        except Exception:
            self._capture_fps = 0.0
        self._update_statusbar()

    def on_poll(self):
        buf = getattr(self.camera, 'buffer', None)
        if buf is None:
            return

        last = None
        try:
            while buf.avail > 0:
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
            img = self.camera.convertQimage(frame)
            if img is None:
                raise ValueError('convertQimage() returned None')
            pix = QPixmap.fromImage(img)
            self.video.setPixmap(pix.scaled(self.video.size(), _KEEP_ASPECT, _FAST_TRANSFORM))

            now = time.perf_counter()
            self._display_frames += 1
            dt = now - self._display_fps_last_t
            if dt >= 5.0:
                self._display_fps = self._display_frames / dt
                self._display_frames = 0
                self._display_fps_last_t = now
                self._update_statusbar()
        except Exception as exc:
            self.video.setText(f'Frame error: {exc}')

    def resizeEvent(self, event):
        pm = self.video.pixmap()
        if pm is not None and not pm.isNull():
            self.video.setPixmap(pm.scaled(self.video.size(), _KEEP_ASPECT, _FAST_TRANSFORM))
        super().resizeEvent(event)

    def closeEvent(self, event):
        try:
            self.camera.close(timeout=2.0)
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 600)
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
