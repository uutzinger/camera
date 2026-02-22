### Dictionary  <!-- omit from toc -->

Either create dictionary in your program or import ti from the config folder e.g. [examples/configs/all_capture_configs.py](examples/configs/all_capture_configs.py)

1) set appropriate resolution and frames per second.
2) determine the exposure and autoexposure settings.
3) consider white balancing.

#### Video Flip <!-- omit from toc -->

| method | name                 | description                                         |
| ------ | -------------------- | --------------------------------------------------- |
| 0      | none                 | Identity (no rotation)                              |
| 1      | clockwise            | Rotate clockwise 90 degrees                         |
| 2      | rotate-180           | Rotate 180 degrees                                  |
| 3      | counterclockwise     | Rotate counter-clockwise 90 degrees                 |
| 4      | horizontal-flip      | Flip horizontally                                   |
| 5      | vertical-flip        | Flip vertically                                     |
| 6      | upper-left-diagonal  | Flip across upper left/lower right diagonal         |
| 7      | upper-right-diagonal | Flip across upper right/lower left diagonal         |
| 8      | automatic            | Select flip method based on image-orientation tag   |

#### Exposure and Autoexposure  <!-- omit from toc -->

The config keys `exposure` and `autoexposure` are interpreted differently depending on the capture backend (OpenCV, GStreamer, Picamera2/libcamera, legacy picamera).

|key|what it does|typical values|
|---|---|---|
|`exposure`|Manual exposure request.|`>0`: time/value (often µs on PiCamera2/libcamera and Jetson nvargus). Some OpenCV webcam drivers use negative “stops-like” values (e.g. `-5` ≈ $2^{-5} = (1/32)$ on some devices; not portable). `0`/`-1`: often means “auto/leave default”, depending on backend.|
|`autoexposure`|Auto-exposure (AE) on/off request.|OpenCV: `-1` keep, `0` manual, `1` auto; many V4L2 cams also accept `0.25` (manual) / `0.75` (auto). Other backends either ignore this key or map it differently.|

Backend details:

- OpenCV (`cv2Capture`):
  - `autoexposure = -1` leaves AE unchanged; `0` requests manual AE mode; `1` requests auto AE mode.
  - Some Linux/V4L2 drivers commonly accept `0.25` (manual) and `0.75` (auto). Windows/macOS backends can differ.
  - `exposure > 0` requests manual exposure and the code attempts to disable AE first. The numeric meaning of `CAP_PROP_EXPOSURE` is driver-specific (some cameras use negative values, others use milliseconds/microseconds-like scales).
  - To inspect what your OpenCV backend/camera supports, run `python3 examples/list_cv2CameraProperties.py`.

- GStreamer (`gCapture`):
  - `gCapture` builds a pipeline and the meaning of “exposure” depends on the selected source element:
    - Jetson `nvarguscamerasrc`: `exposure` is treated as microseconds (converted internally) and applied via `exposuretimerange` when supported.
    - `libcamerasrc`: exposure/AE are controlled via libcamera controls, but the exact property interface is element-version specific. Use `gst-inspect-1.0 libcamerasrc` and set the element’s control-related properties via `gst_source_props_str`, or provide a complete custom source with `gst_source_str`.
    - `v4l2src`: exposure is typically controlled via V4L2 controls (recommended: `v4l2-ctl` from `v4l-utils`). Some GStreamer builds expose an `extra-controls`-style property on `v4l2src`, but it is not consistent across platforms/drivers.
    - Windows `ksvideosrc` / `dshowvideosrc`: many cameras do not expose manual exposure controls through these elements. If the element exposes a property for it, you can set it via `gst_source_props_str`; otherwise use vendor tools / OS camera settings (or configure exposure through a separate API).
  - Use `gst-inspect-1.0 <element>` to discover supported properties and set them via `gst_source_props_str` or a fully custom `gst_source_str`.

- FLIR/Teledyne Spinnaker (`spinCapture`, PySpin):
  - `exposure` is in microseconds and maps to `ExposureTime` (requires `ExposureAuto` off and `ExposureMode=Timed`).
  - `autoexposure`: `0` off, `1` on (continuous/once depending on camera settings).

- Raspberry Pi Picamera2/libcamera (`piCamera2Capture`):
  - `exposure` is interpreted as microseconds and mapped to libcamera `ExposureTime`.
  - `autoexposure` maps to libcamera `AeEnable` (`>0` enables, `0` disables, `-1` leaves unchanged).

- Raspberry Pi legacy picamera (`piCapture`):
  - Exposure is controlled via `shutter_speed` in microseconds.
  - In this backend, `exposure <= 0` uses automatic exposure (`exposure_mode='auto'`); `exposure > 0` requests manual exposure (`exposure_mode='off'` plus `shutter_speed`).

Maximum usable exposure time is typically limited by frame interval. Roughly, max exposure is about $\frac{10^6}{\mathrm{fps}}$ microseconds (camera/driver may clamp differently).
