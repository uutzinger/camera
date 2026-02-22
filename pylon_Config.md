### Basler pylon (pypylon) supported config keys

#### Stream modes:
  - `max_fps`
    - `isp_enable=False`
    - `gain_auto=0, gain=1.0`
    - `autoexposure=0, exposure_mode='timed'`
    - `acquisition_mode='continuous'`
    - `acquisition_frame_rate_enable=True`
    - `gamma=1.0, gamma_enable=False`
    - `balance_white_auto=0`
  - `auto`
    - `isp_enable=True`
    - `autoexposure=1, gain_auto=1, balance_white_auto=1`
    - `acquisition_mode='continuous'`
    - `gamma_enable=True`
  - `default`
    - no baseline overrides; rely on camera/driver defaults.

#### Core capture/output:
- `camera_res: (w, h)`        Capture size
- `output_res: (w, h)`        Optional CPU resize target (`(-1, -1)` means no resize)
- `fps: float|int`            Requested framerate
- `flip: int`                 0..7 (same enum as cv2Capture)
- `stream_policy: str`        `'default' | 'max_fps' | 'auto'`
- `buffersize: int`           FrameBuffer capacity (default 32)
- `buffer_overwrite: bool`    Overwrite oldest frames when full
- `capture_timeout_ms: int`   `RetrieveResult` timeout in ms

#### Sensor / format controls:
- `exposure: float|int`       Exposure time in microseconds
- `autoexposure: int`         -1 unchanged, 0 off, 1 on
- `isp_enable: bool`          Camera-side processing best-effort (model dependent)
- `gain: float`               Manual gain
- `gain_auto: int|bool|str`   -1 unchanged, 0 off, 1 on, or camera mode string
- `binning: int|tuple`        Sensor binning (h,v)
- `offset: int|tuple`         ROI offset (x,y)
- `adc: int`                  Sensor bit depth best-effort (model dependent)
- `pixel_format: str`         Pixel format override (e.g. `Mono8`, `Mono16`, `BayerRG8`)
- `convert_format: str`       Conversion target (`native`, `Mono8`, `Mono16`, `BGR8`, `RGB8`)
- `acquisition_mode: str`     Acquisition mode override (typically `continuous`)
- `exposure_mode: str`        Exposure mode override (typically `timed`)
- `acquisition_frame_rate_enable: bool`
- `gamma: float`              Intensity gamma correction
- `gamma_enable: bool`
- `balance_white_auto: int|bool|str` automatic/manual white balance mode

#### Trigger / IO:
- `trigin: int`               `-2` leave default, `-1` software trigger, else line index (`1 => Line1`)
- `trigout: int`              `-1` disable, else line index (`1 => Line1`)
- `ttlinv: bool`              Invert trigger polarity

#### Notes:
- Stream policy is applied first; explicit config keys override it.
- `convert_format` uses CPU conversion; `output_res` and `flip` use OpenCV.
- GenICam features differ between camera models and interfaces. Unsupported keys are logged and ignored.
