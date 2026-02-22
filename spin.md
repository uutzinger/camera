## Spin Wrapper (PySpin / Spinnaker)

This repository provides a non-threaded Spinnaker core and two wrappers that
expose a consistent API across non-Qt and Qt environments.

### Classes

#### `SpinCore` (non-threaded core)
Primary responsibility: normalize configuration, open/configure the camera,
provide capture with a preallocated FrameBuffer, and expose PySpin controls.

Methods:
- `open_cam() -> bool`
- `close_cam() -> None`
- `capture_array() -> (frame, ts_ms)`
- `trigger_software() -> bool`
- `get_control(name: str) -> Any`
- `set_controls(controls: dict) -> bool`
- `get_supported_main_color_formats() -> list[str]`
- `get_supported_raw_color_formats() -> list[str]`
- `get_supported_raw_options() -> list[dict]`
- `get_supported_main_options() -> list[dict]`
- `log_stream_options() -> None`

Properties (read/write unless noted):
- `cam_open` (bool, read-only)
- `buffer` (FrameBuffer, read-only)
- `metadata` (dict, read-only)
- `width`, `height`, `resolution`, `size`
- `exposure`, `autoexposure`, `fps`, `flip`, `stream_policy`
- `isp_enable`, `gain`, `gain_auto`
- `acquisition_mode`, `exposure_mode`, `acquisition_frame_rate_enable`
- `gamma`, `gamma_enable`, `balance_white_auto`
- `binning`, `offset`, `adc`, `pixel_format`
- `trigin`, `trigout`, `ttlinv`
- `convert_format` (target conversion format, default `native`)

Notes on processing:
- Camera-side ISP: `isp_enable` controls `IspEnable` when supported.
- CPU conversion: `convert_format` uses `PySpin.ImageProcessor.Convert(...)`.
- CPU geometry: `output_res` and `flip` are applied via OpenCV.

#### `spinCapture` (threaded, non-Qt)
Methods:
- `start()`, `stop(timeout: float | None = 2.0)`, `join(timeout: float | None = None)`
- `close()`
- `log_stream_options()`
- `set_exposure_us(exposure_us: int)`
- `set_auto_exposure(enabled: bool)`
- `set_framerate(fps: float)`
- `set_aemeteringmode(mode)` (not supported by PySpin; logs warning)
- `set_auto_wb(enabled: bool)` (maps to `balance_white_auto`)
- `set_awbmode(mode)` (maps to `balance_white_auto` best-effort)
- `set_flip(flip: int)`
- `set_resolution(res: tuple[int, int])`
- `set_stream_policy(policy: str)`
- `set_isp_enable(enabled: bool)`
- `set_gain(gain: float)`
- `set_gain_auto(enabled_or_mode)`
- `set_acquisition_mode(mode: str)`
- `set_exposure_mode(mode: str)`
- `set_acquisition_frame_rate_enable(enabled: bool)`
- `set_gamma(value: float)`
- `set_gamma_enable(enabled: bool)`
- `set_balance_white_auto(enabled_or_mode)`
- `set_binning(val)`
- `set_offset(val)`
- `set_adc(bits: int)`
- `set_pixel_format(fmt: str)`
- `set_convert_format(fmt: str)`
- `set_trigin(line: int)`
- `set_trigout(line: int)`
- `set_ttlinv(inverted: bool)`
- `trigger_software() -> bool`
- `get_supported_main_formats()`, `get_supported_raw_formats()`
- `get_supported_main_options()`, `get_supported_raw_options()`
- `get_control(name: str)`
- `__getattr__` delegates to `SpinCore`

Attributes:
- `buffer` (FrameBuffer)
- `capture` (alias of `buffer`, not a Queue)
- `log` (Queue of `(level, message)`)
- `measured_fps` (float)

#### `spinCaptureQt` (threaded, Qt)
Same API as `spinCapture`, plus:
- Qt signals: `stats(float)`, `log(int, str)`, `opened()`, `started()`, `stopped()`
- `convertQimage(frame) -> QImage | None`
- `pause()`, `resume()`, `is_capturing`, `is_running`

### Notes
- Stream policy is applied first, then explicit overrides from config/setters.
- Disabling ISP does not disable CPU conversion or OpenCV resize/flip.
