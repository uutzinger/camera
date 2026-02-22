## RTSP Wrapper

This repository provides a non-threaded RTSP core and two thin wrappers that
expose a consistent API across non-Qt and Qt environments.

### Classes

#### `RtspCore` (non-threaded core)
Primary responsibility: open/configure the RTSP stream, capture frames, and
apply optional software conversions (resize/flip).

Methods:
- `open_cam() -> bool`
- `close_cam() -> None`
- `capture_array() -> (frame, ts_ms)`
- `get_control(name: str) -> Any`
- `log_stream_options() -> None`
- `set_output_res(res: tuple[int, int]) -> None`
- `set_flip(flip: int) -> None`

Properties (read/write unless noted):
- `cam_open` (bool, read-only)
- `buffer` (FrameBuffer, read-only)
- `backend` (str | None, read-only)

#### `rtspCapture` (threaded, non-Qt)
Methods:
- `start()`, `stop(timeout: float | None = 2.0)`, `join(timeout: float | None = None)`
- `close(timeout: float | None = 2.0)`
- `log_stream_options()`
- `set_output_res(res: tuple[int, int])`
- `set_resolution(res: tuple[int, int])` (alias of `set_output_res`)
- `set_flip(flip: int)`
- `get_control(name: str)`
- `__getattr__` delegates to `RtspCore`

Attributes:
- `buffer` (FrameBuffer)
- `capture` (alias of `buffer`, not a Queue)
- `log` (Queue of `(level, message)`)
- `measured_fps` (float)

#### `rtspCaptureQt` (threaded, Qt)
Same API as `rtspCapture`, plus:
- Qt signals: `stats(float)`, `log(int, str)`, `opened()`, `started()`, `stopped()`
- `convertQimage(frame) -> QImage | None`
