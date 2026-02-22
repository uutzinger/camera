## GStreamer Wrapper (gCapture)

This repository provides a threaded GStreamer capture wrapper focused on custom
pipeline sources and low-latency appsink delivery.

### Classes

#### `gCapture` (threaded, non-Qt)
Primary responsibility: build a source-aware GStreamer pipeline, capture frames,
and deliver them through a queue.

Methods:
- `start()`, `stop()`, `run()`, `update()`
- `open_cam() -> None`
- `close_cam() -> None`
- `gstreamer_pipeline(...) -> str`

Backend behavior:
- Preferred: GI/PyGObject (`Gst` + appsink)
- Fallback: OpenCV `VideoCapture(..., CAP_GSTREAMER)`

Attributes:
- `capture` (`Queue[(ts_ms, frame)]`)
- `log` (`Queue[(level, message)]`)
- `cam_open` (`bool`)
- `measured_fps` (`float`)

#### `gCaptureQt` (threaded, Qt)
Qt wrapper that composes `gCapture` and forwards queue-driven capture/logging
through Qt signals.

Methods:
- `start()`, `stop(timeout: float | None = 2.0)`, `join(timeout: float | None = None)`
- `close(timeout: float | None = 2.0)`
- `convertQimage(frame) -> QImage | None`

Signals:
- `stats(float)`, `log(int, str)`, `opened()`, `started()`, `stopped()`

Attributes:
- `capture` (`Queue[(ts_ms, frame)]`)
- `log_queue` (`Queue[(level, message)]`)
- `cam_open` (`bool`)
- `measured_fps` (`float`)

Notes:
- `gCapture` is Queue-based (legacy style), not FrameBuffer-based.
- `nanoCapture` is now a Jetson-focused wrapper around `gCapture` defaults.

Source-specific references:
- Jetson NVArgus: `gCapture_nvarguscam.md`
- libcamera: `gCapture_libcameracam.md`
- Linux V4L2: `gCapture_v4l2cam.md`
- Windows KS: `gCapture_ksvideocam.md`
- Windows DirectShow: `gCapture_dshowcam.md`

Examples:
- Non-Qt display: `examples/gcapture_display.py`
- Qt display: `examples/qt_gcapture_display.py`
- Capture + RTP send: `examples/gcapture_display_send2rtp.py`
