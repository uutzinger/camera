### Supported config keys

Core capture/output:
- `camera_res`: requested camera frame size `(w, h)`. This should match the formats supported by the camera.
- `output_res: (w, h)` Prefer ISP scaling for main stram; when RAWs stream is used this may trigger CPU resize, `(-1, -1)` for “same as camera_res”
- `fps: float|int` requested framerate, this should match the rates supported by the camera.
- `low_latency: bool` reduce buffering if supported
- `buffersize: int` frameBuffer capacity (default 32)
- `buffer_overwrite: bool` overwrite oldest frames when full
- `test_pattern: bool|str` sSynthetic frame mode for testing

Stream selection:
- `mode`: `'main'` (processed, full-FOV stream) or `'raw'` (raw sensor window, potentially cropped FOV).
- `stream_policy: 'default' | 'maximize_fps_no_crop' | 'maximize_fps_with_crop' | 'maximize_fov'`

Formats:
- `format: str` legacy/combined request, supports regular `BGR888`, fourcc (e.g. `BGR3`, `YUY2`) and raw Bayer format `SRGGB8`
- convert_format: str` target conversion format (default `BGR888`)

Controls:
- `exposure: int|float` Manual exposure in microseconds; <=0 leaves AE enabled
- `autoexposure: int`-1 unchanged, 0 AE off, 1 AE on
- `aemeteringmode: int` (0=CentreWeighted, 1=Spot, 2=Matrix) or string (`"center"`, `"spot"`, `"matrix"`).
- `autowb: int` -1 unchanged, 0 AWB off, 1 AWB on
- `awbmode: int|str` 0..5 or `auto`/`tungsten`/`fluorescent`/`indoor`/`daylight`/`cloudy`
- `flip: int` 0..7 (same enum as cv2Capture)

Low-level/advanced:
- `buffer_count: int|None` requested libcamera buffer_count, 
- `hw_transform: bool` attempt libcamera Transform for flip/rotation

### Notes
- Camera is opened in `__init__` with the provided configs. `start()/stop()` only
  control frame capture, not camera open/close.
