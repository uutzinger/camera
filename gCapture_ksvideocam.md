## ksvideosrc Reference (Windows)

Options for `ksvideo` source in `gCapture`.

Use these options through:
- `configs["gst_source"] = "ksvideo"` (or `"ksvideosrc"`)
- `configs["gst_source_props_str"] = "key=value ..."`

Example:
```python
configs = {
    "gst_source": "ksvideo",
    "gst_source_props_str": "device-index=0 do-stats=true",
    "camera_res": (1280, 720),
    "fps": 30,
}
```

## Commonly used properties

- `device-index` (int): select capture device by index.
- `device-name` (string, read-only): selected device name.
- `device-path` (string, read-only): selected device path.
- `do-stats` (bool): enable source statistics.
- `enable-quirks` (bool): apply known compatibility quirks.
- `fps` (int/float, read-only): measured framerate.

## Notes for this repository

- `gCapture` auto-selects `ksvideosrc` first on Windows when available.
- If your GStreamer build does not include `ksvideosrc`, `gCapture` tries
  `dshowvideosrc` next.
- For fully custom Windows source strings, use `gst_source_str`.
- `gCapture` currently maps only source selection, optional `device-index`, and
  caps (`width/height/fps`) for `ksvideosrc`.
- Exposure/white-balance are not mapped via dedicated config keys in current
  code; if your plugin build exposes such properties, pass them via
  `gst_source_props_str`.

## Discoverability / troubleshooting

- List video sources:
  - `gst-device-monitor-1.0 Video/Source`
- Show element properties:
  - `gst-inspect-1.0 ksvideosrc`

## Sources

- GStreamer Windows KS plugin docs (`winks`):
  - https://gstreamer.freedesktop.org/documentation/winks/
