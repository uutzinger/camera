### GStreamer supported config keys

Core capture/output:
- `camera_res: (w, h)`            Capture size
- `output_res: (w, h)`            Optional output size (uses videoscale)
- `fps: float|int`                Requested framerate
- `exposure: float`               Source-dependent, typically microseconds
- `flip: int`                     0..7 (same enum as cv2Capture)

Source selection / customization:
- `gst_source: str`               Source selector (`auto`, `libcamera`, `nvargus`, `v4l2`, `ksvideo`, `dshow`)
- `source: str`                   Alias for `gst_source`
- `gst_source_str: str`           Full custom source snippet
- `gst_source_pipeline: str`      Alias for `gst_source_str`
- `gst_source_props_str: str`     Extra source-element properties appended after element name

V4L2 convenience:
- `device: str`                   V4L2 device path (for `v4l2src`), e.g. `/dev/video0`
- `v4l2_device: str`              Alias for `device`
- `v4l2_format: str`              V4L2 source format hint (`raw`, `mjpg`, `mjpeg`, `jpeg`)
- `pixel_format: str`             Alias used with V4L2 selection path

### Notes
- Preferred backend uses GI/PyGObject (`Gst` + appsink).
- Fallback backend uses OpenCV `CAP_GSTREAMER` when GI is unavailable or pipeline init fails.
- OpenCV fallback requires OpenCV built with GStreamer support.
- Same config keys apply to `gCaptureQt`.
- Queue size is configured via the constructor parameter `queue_size` (not a config key).
- Current source-specific behavior in code:
  - `nvargus`: `exposure > 0` maps to `exposuretimerange="<ns> <ns>"` and `aelock=true` (if supported).
  - `v4l2`: code sets `device`, `io-mode=2`, caps (`width/height/framerate`) and optional MJPEG path.
  - `libcamera`, `ksvideo`, `dshow`: code sets source + caps and optional device index (Windows).
  - Exposure/white-balance for non-`nvargus` sources are not mapped to dedicated config keys; pass them through with `gst_source_props_str`.

### Source documentation
- `nvargus` / `nvarguscamerasrc`: [gCapture_nvarguscam.md](gCapture_nvarguscam.md)
- `libcamera` / `libcamerasrc`: [gCapture_libcameracam.md](gCapture_libcameracam.md)
- `v4l2` / `v4l2src`: [gCapture_v4l2cam.md](gCapture_v4l2cam.md)
- `ksvideo` / `ksvideosrc`: [gCapture_ksvideocam.md](gCapture_ksvideocam.md)
- `dshow` / `dshowvideosrc`: [gCapture_dshowcam.md](gCapture_dshowcam.md)
