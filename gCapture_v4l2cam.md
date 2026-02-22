## v4l2src Reference (Linux)

Options for `v4l2` source in `gCapture`.

Use these options through:
- `configs["gst_source"] = "v4l2"`
- `configs["device"] = "/dev/video0"` (or `v4l2_device`)
- `configs["gst_source_props_str"] = "key=value ..."`

Example:
```python
configs = {
    "gst_source": "v4l2",
    "device": "/dev/video2",
    "gst_source_props_str": "io-mode=2 do-timestamp=true",
    "camera_res": (1280, 720),
    "fps": 30,
}
```

## Commonly used properties

- `device` (string): V4L2 device path (default `/dev/video0`).
- `io-mode` (enum): capture I/O mode.
  - `0` auto
  - `1` rw
  - `2` mmap
  - `3` userptr
  - `4` dmabuf
  - `5` dmabuf-import
- `do-timestamp` (bool): apply current stream time to buffers.
- `num-buffers` (int): number of buffers before EOS (`-1` unlimited).
- `blocksize` (uint): read size per buffer.
- `extra-controls` (GstStructure): pass additional V4L2 controls.
- `brightness`, `contrast`, `hue`, `saturation` (int): image controls.
- `pixel-aspect-ratio` (string): override reported pixel aspect ratio.

## Notes for this repository

- `configs["v4l2_format"]` (or `pixel_format`) controls the `gCapture` source
  branch:
  - `raw` (default): `video/x-raw ...`
  - `mjpg` / `mjpeg` / `jpeg`: `image/jpeg ... ! jpegdec`
- MJPEG support and max FPS depend on the camera driver and decoder plugins.
- `gCapture` currently does not map exposure/WB to dedicated config keys for
  `v4l2src`. Use `gst_source_props_str` (and often `extra-controls`) instead.

Example pattern for common UVC controls (driver dependent):
```python
configs = {
    "gst_source": "v4l2",
    "device": "/dev/video0",
    "gst_source_props_str": (
        'do-timestamp=true '
        'extra-controls="c,exposure_auto=1,exposure_absolute=200,'
        'white_balance_temperature_auto=0,white_balance_temperature=4500"'
    ),
}
```

## Discoverability / troubleshooting

- Show source capabilities:
  - `v4l2-ctl --device=/dev/video0 --list-formats-ext`
- Show GStreamer properties/caps:
  - `gst-inspect-1.0 v4l2src`

## Sources

- Official GStreamer v4l2src docs:
  - https://gstreamer.freedesktop.org/documentation/video4linux2/v4l2src.html
