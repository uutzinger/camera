### Spinnaker (PySpin) supported config keys

#### Stream modes:
  - max-fps
    - isp_enable=False
    - gain_auto=0, gain=1.0
    - autoexposure=0, exposure_mode='timed'
    - acquisition_mode='continuous'
    - acquisition_frame_rate_enable=True
    - gamma=1.0, gamma_enable=False
    - balance_white_auto=0
  - 'auto':
    - isp_enable=True
    - autoexposure=1, gain_auto=1, balance_white_auto=1
    - acquisition_mode='continuous'
    - gamma_enable=True
  - 'default':
    - no baseline overrides; rely on camera/driver defaults.

#### Core capture/output:
- `camera_res: (w, h)`        Capture size
- `output_res: (w, h)`        Optional CPU resize target (`(-1, -1)` means no resize)
- `fps: float|int`            Requested framerate
- `flip: int`                 0..7 (same enum as cv2Capture)
- `stream_policy: str`        `'default' | 'max_fps' | 'auto'`
- `buffersize: int`           FrameBuffer capacity (default 32)
- `buffer_overwrite: bool`    Overwrite oldest frames when full
- `capture_timeout_ms: int`   PySpin GetNextImage timeout in ms

#### Sensor / format controls:
- `exposure: float|int`       Exposure time in microseconds
- `autoexposure: int`         -1 unchanged, 0 off, 1 on
- `isp_enable: bool`          Enable/disable camera ISP when available
- `gain: float`               Manual gain
- `gain_auto: int|bool`       -1 unchanged, 0 off, 1 on (or `'once'`)
- `binning: int|tuple`        Sensor binning (h,v)
- `offset: int|tuple`         ROI offset (x,y)
- `adc: int`                  ADC bit depth (8,10,12,14)
- `pixel_format: str`         Pixel format override (e.g. `Mono8`, `Mono16`, `BayerRG8`)
- `convert_format: str`       Conversion target (`native`, `Mono8`, `Mono16`, `BGR8`, `RGB8`)
- `color_processing: str`     ImageProcessor algorithm (default `HQ_LINEAR`)
- `acquisition_mode: str`     Acquisition mode override (e.g. `continuous` | `singleframe` | `multiframe`)
- `exposure_mode: str`        Exposure mode override (e.g. `timed` | `trigger_width`) with timed camera records image during exposure time.
- `acquisition_frame_rate_enable: bool` attempts to maintain fps regardless of exposure time
- `gamma: float`              Intensity gamma correction
- `gamma_enable: bool`
- `balance_white_auto: int|bool` automatic or manual white balancing

#### Trigger / IO:
- `trigin: int`               -1 software trigger, else line index (0..3)
- `trigout: int`              -1 disable, else line index (1..3 typical)
- `ttlinv: bool`              Invert trigger polarity

#### Notes:
- Stream policy is applied first; explicit config keys override it.
- `convert_format` uses CPU conversion; `output_res` and `flip` use OpenCV.
