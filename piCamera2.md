## Picamera2 Wrapper (libcamera)

This repository provides a Picamera2 core and two wrappers that expose a
consistent API across non-Qt and Qt environments.

### Classes

#### `PiCamera2Core` (non-threaded core)
Primary responsibility: normalize configuration, open/configure the camera,
capture frames (main or raw), and provide control/metadata helpers.

Methods:
- `open_cam() -> bool`
- `close_cam() -> None`
- `capture_array() -> (frame, ts_ms)`
- `get_control(name: str) -> Any`
- `set_controls(controls: dict) -> bool`
- `get_supported_main_color_formats() -> list[str]`
- `get_supported_raw_color_formats() -> list[str]`
- `get_supported_main_options() -> list[dict]`
- `get_supported_raw_options() -> list[dict]`
- `log_stream_options() -> None`

Properties (read/write unless noted):
- `cam_open` (bool, read-only)
- `buffer` (FrameBuffer, read-only)
- `metadata` (dict, read-only)
- `width`, `height`, `resolution`, `size`
- `exposure`, `gain`, `autoexposure`, `aemeteringmode`, `aeexposuremode`
- `autowb`, `awbmode`, `wbtemperature`, `colourgains`
- `afmode`, `lensposition`
- `brightness`, `contrast`, `saturation`, `sharpness`, `noisereductionmode`
- `fps`
- `convert_format` (target conversion format, default `BGR888`)

#### `piCamera2Capture` (threaded, non-Qt)
Methods:
- `start()`, `stop(timeout: float | None = 2.0)`, `join(timeout: float | None = None)`
- `close()`
- `log_stream_options()`
- `set_exposure_us(exposure_us: int)`
- `set_auto_exposure(enabled: bool)`
- `set_framerate(fps: float)`
- `set_aemeteringmode(mode)`
- `set_auto_wb(enabled: bool)`
- `set_awbmode(mode)`
- `set_flip(flip: int)`
- `set_resolution(res: tuple[int, int])`
- `get_supported_main_formats()`, `get_supported_raw_formats()`
- `get_supported_main_options()`, `get_supported_raw_options()`
- `get_control(name: str)`
- `__getattr__` delegates to `PiCamera2Core`

Attributes:
- `buffer` (FrameBuffer)
- `capture` (alias of `buffer`, not a Queue)
- `log` (Queue of `(level, message)`)
- `measured_fps` (float)

#### `piCamera2CaptureQt` (threaded, Qt)
Same API as `piCamera2Capture`, plus:
- Qt signals: `stats(float)`, `log(int, str)`, `opened()`, `started()`, `stopped()`
- `convertQimage(frame) -> QImage | None`

