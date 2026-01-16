"""Qt5/Qt6 example: Raspberry Pi Picamera2 display with Start/Stop.

This is the Qt counterpart to examples/picamera2_capture_display.py.

- Uses camera.capture.picamera2captureQt.piCamera2CaptureQt
- Displays frames in a Qt window
- Provides a Start/Stop toggle button

Run:
    python3 examples/qt_picamera2_capture_display.py

Notes:
- This is intended for Raspberry Pi OS with `picamera2` installed.
- If running headless, you still need a working Qt display backend.
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


os.environ.setdefault('LIBCAMERA_LOG_LEVELS', '*:3')  # 3=ERROR


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('PiCamera2 Qt Display')

        # Match picamera2_capture_display defaults (can be edited)
        self.configs = {
            ################################################################
            # Picamera2 capture configuration
            #
            # List camera properties with:
            #     examples/list_Picamera2Properties.py
            ################################################################
            # Capture mode:
            #   'main' -> full-FOV processed stream (BGR/YUV), scaled to 'camera_res' (libcamera scales)
            #   'raw'  -> high-FPS raw sensor window (exact sensor mode only), cropped FOV
            'mode'            : 'main',
            'camera_res'      : (640, 480),     # requested main stream size (w, h)
            'exposure'        : 0,              # microseconds, 0/-1 for auto
            'fps'             : 60,             # requested capture frame rate
            'autoexposure'    : 1,              # -1 leave unchanged, 0 AE off, 1 AE on
            'aemeteringmode'  : 'center',       # int or 'center'|'spot'|'matrix'
            'autowb'          : 1,              # -1 leave unchanged, 0 AWB off, 1 AWB on
            'awbmode'         : 'auto',         # int or friendly string
            # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
            # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
            'format'          : 'BGR3',
            'stream_policy'   : 'default',      # 'maximize_fov', 'maximize_fps', 'default'
            'low_latency'     : True,           # low_latency=True prefers size-1 buffer (latest frame)
            'buffersize'      : 4,              # FrameBuffer capacity override (wrapper-level)
            'output_res'      : (-1, -1),       # (-1,-1): output == input; else libcamera scales main
            'flip'            : 0,              # 0=norotation
            'displayfps'      : 30,             # consumer-side display throttle
            'test_pattern'    : 'gradient',     # enable synthetic frames (bypass Picamera2/libcamera)
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

        self.camera = piCamera2CaptureQt(self.configs, camera_num=0)
        self.camera.stats.connect(self.on_stats)
        self.camera.log.connect(self.on_log)
        self.camera.started.connect(self.on_started)
        self.camera.stopped.connect(self.on_stopped)

        self._poll_timer = QTimer(self)
        interval_ms = 0 if self._display_interval <= 0.0 else max(1, int(self._display_interval * 1000.0))
        self._poll_timer.setInterval(interval_ms)
        self._poll_timer.timeout.connect(self.on_poll)

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

        self._logger = logging.getLogger('picamera2_capture_qt_display')

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
            self._logger.info("%s", message)

    def on_stats(self, fps: float):
        try:
            self._capture_fps = float(fps)
        except Exception:
            self._capture_fps = 0.0
        self._update_statusbar()

    def on_poll(self):
        buf = getattr(self.camera, "buffer", None)
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
            # Keep aspect ratio and fit the label
            self.video.setPixmap(pix.scaled(self.video.size(), _KEEP_ASPECT, _FAST_TRANSFORM))

            # Display FPS: count successful GUI updates and update periodically.
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
        # Re-scale current pixmap when resizing
        pm = self.video.pixmap()
        if pm is not None and not pm.isNull():
            self.video.setPixmap(pm.scaled(self.video.size(), _KEEP_ASPECT, _FAST_TRANSFORM))
        super().resizeEvent(event)

    def closeEvent(self, event):
        # Ensure capture thread shuts down
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


if __name__ == "__main__":
    raise SystemExit(main())
