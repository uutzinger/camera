## libcamerasrc Reference (Linux / libcamera)

Options for `libcamera` source in `gCapture`.

Use these options through:
- `configs["gst_source"] = "libcamera"`
- `configs["gst_source_props_str"] = "key=value ..."`

Example:
```python
configs = {
    "gst_source": "libcamera",
    "gst_source_props_str": 'camera-name="/base/soc/i2c0mux/i2c@1/imx219@10"',
}
```

## Commonly used properties

- `camera-name` (string): select a specific libcamera camera by name/path.
- `auto-focus-mode` (enum/int, version-dependent): autofocus behavior where supported.
- `list-cameras` (bool, version-dependent): logs available cameras on startup.

## Pipeline notes

- In this repository, `gCapture` appends:
  - `video/x-raw,width=...,height=...,framerate=...`
  - `format=NV12` by default for `libcamerasrc`.
- Current `gCapture` code does not map dedicated exposure/white-balance keys for
  `libcamera`. Those controls (if supported by your `libcamerasrc` build) must
  be passed via `gst_source_props_str` or a full `gst_source_str`.
- Some advanced controls are exposed as libcamera controls rather than stable
  GStreamer element properties.

Example pattern (property names are plugin-version dependent):
```python
configs = {
    "gst_source": "libcamera",
    "gst_source_props_str": 'camera-controls="controls,ExposureTime=10000,AwbEnable=false"',
}
```

## Discoverability / troubleshooting

- List cameras:
  - `gst-device-monitor-1.0 Video/Source`
- Show available properties on your system:
  - `gst-inspect-1.0 libcamerasrc`

## Sources

- libcamera + GStreamer quickstart:
  - https://libcamera.org/getting-started.html
- Recent libcamerasrc property additions (camera-name / list-cameras / autofocus):
  - https://patchwork.libcamera.org/patch/17631/
  - https://patchwork.libcamera.org/patch/19013/
