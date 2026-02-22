## dshowvideosrc Reference (Windows)

Options for `dshow` source in `gCapture`.

Use these options through:
- `configs["gst_source"] = "dshow"` (or `"dshowvideosrc"`)
- `configs["gst_source_props_str"] = "key=value ..."`

Example:
```python
configs = {
    "gst_source": "dshow",
    "gst_source_props_str": 'device-name="USB Camera"',
    "camera_res": (1280, 720),
    "fps": 30,
}
```

## Commonly used properties

- `device` (string): DirectShow moniker/path of capture device.
- `device-name` (string): select capture device by friendly name.
- `device-index` (int): select capture device by index.

## Notes for this repository

- `gCapture` uses `dshowvideosrc` as a Windows fallback when `ksvideosrc` is
  not present.
- Device indexing behavior may vary by installed DirectShow filters.
- For unusual devices/filter graphs, use `gst_source_str` with a fully custom
  source section.
- `gCapture` currently maps only source selection, optional `device-index`, and
  caps (`width/height/fps`) for `dshowvideosrc`.
- Exposure/white-balance are not mapped to dedicated config keys in current
  code; use source-specific properties via `gst_source_props_str` when
  available.

## Discoverability / troubleshooting

- List video sources:
  - `gst-device-monitor-1.0 Video/Source`
- Show element properties:
  - `gst-inspect-1.0 dshowvideosrc`

## Sources

- Official GStreamer dshowvideosrc docs:
  - https://gstreamer.freedesktop.org/documentation/directshow/dshowvideosrc.html
