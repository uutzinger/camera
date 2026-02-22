## OpenCV Wrapper (cv2)

This repository provides a non-threaded OpenCV core and two thin wrappers that
expose a consistent API across non-Qt and Qt environments.

`probeCameras(numcams)` attempts to find all camera available
`findCamera(numcams = 10, fourccSig = "\x16\x00\x00\x00", widthSig=640, heightSig=480)` returns the index of the camera matching the signature.

### Classes

#### `cv2Capture` (threaded, non-Qt)
Methods:
- `start()`, `stop(timeout: float | None = 2.0)`, `join(timeout: float | None = None)`
- `close(timeout: float | None = 2.0)`
- `log_stream_options()`
- `set_exposure_us(exposure_us: int)`
- `set_auto_exposure(enabled: bool)`
- `set_framerate(fps: float)`
- `set_aemeteringmode(mode)` (not supported by OpenCV backends; logs warning)
- `set_auto_wb(enabled: bool)`
- `set_awbmode(mode)` (not supported by OpenCV backends; logs warning)
- `set_flip(flip: int)`
- `set_resolution(res: tuple[int, int])`
- `get_supported_main_formats()`, `get_supported_raw_formats()`
- `get_supported_main_options()`, `get_supported_raw_options()`
- `get_control(name: str)`
- `__getattr__` delegates to `cv2Core`

Attributes:
- `buffer` (FrameBuffer)
- `capture` (alias of `buffer`, not a Queue)
- `log` (Queue of `(level, message)`)
- `measured_fps` (float)

#### `cv2CaptureQt` (threaded, Qt)
Same API as `cv2Capture`, plus:
- Qt signals: `stats(float)`, `log(int, str)`, `opened()`, `started()`, `stopped()`
- `convertQimage(frame) -> QImage | None`

#### `cv2Core` (non-threaded core)
Primary responsibility: open/configure the camera, capture frames, and apply
optional software conversions (resize/flip/color conversion). User should not reach to core.

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
- `exposure`, `autoexposure`
- `gain`, `wbtemperature`, `autowb`
- `fps`, `fourcc`, `buffer_count`, `flip`
- `convert_format` (target conversion format, default `BGR888`)

### Notes
- By default OpenCV delivers BGR frames. Software conversion runs only when
  `convert_rgb=False` or when `convert_format` requests RGB.
- `set_aemeteringmode` and `set_awbmode` are no-ops on OpenCV backends.
