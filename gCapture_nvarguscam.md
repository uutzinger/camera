## nvarguscamerasrc Reference (Jetson)

Options for `nvargus` that is accessible in `gCapture` or `nanoCapture`.

Use these options through:
- `configs['gst_source'] = 'nvargus'`
- `configs['gst_source_props_str'] = 'key=value ...'`

Example:
```python
configs = {
    "gst_source": "nvargus",
    "gst_source_props_str": 'aelock=true exposuretimerange="120000 120000" wbmode=1',
}
```

## Commonly used properties

- `sensor-id` (int, default `0`): camera sensor index.
- `sensor-mode` (int, default `-1`): select sensor mode; `-1` auto/best match.
- `exposuretimerange` (string): nanoseconds range, e.g. `"34000 358733000"`.
- `aelock` (bool): lock/unlock auto-exposure.
- `awblock` (bool): lock/unlock auto-white-balance.
- `wbmode` (enum int): white-balance mode.
- `aeantibanding` (enum int): anti-banding mode.
- `exposurecompensation` (float): exposure compensation in `[-2, 2]`.
- `gainrange` (string): sensor gain range, e.g. `"1 16"`.
- `ispdigitalgainrange` (string): ISP digital gain range, e.g. `"1 8"`.
- `tnr-mode` (enum int): temporal-noise-reduction mode.
- `tnr-strength` (float): temporal-noise-reduction strength.
- `ee-mode` (enum int): edge-enhancement mode.
- `ee-strength` (float): edge-enhancement strength.
- `saturation` (float): saturation adjustment.
- `do-timestamp` (bool): stamp buffers with stream time.
- `num-buffers` (int): number of buffers before EOS (`-1` unlimited).
- `timeout` (uint): capture timeout in seconds.
- `blocksize` (uint): per-buffer read size.
- `bufapi-version` (bool): enable new buffer API (JetPack-dependent).

## Enum values

`wbmode`:
- `0` off
- `1` auto
- `2` incandescent
- `3` fluorescent
- `4` warm-fluorescent
- `5` daylight
- `6` cloudy-daylight
- `7` twilight
- `8` shade
- `9` manual

`tnr-mode`:
- `0` off
- `1` fast
- `2` high quality

`ee-mode`:
- `0` off
- `1` fast
- `2` high quality

`aeantibanding`:
- `0` off
- `1` auto
- `2` 50Hz
- `3` 60Hz

## Notes

- Property availability varies by JetPack/L4T and plugin version.
- Current `gCapture` convenience mapping:
  - if `configs["exposure"] > 0`, it attempts to set
    `exposuretimerange="<ns> <ns>"` and `aelock=true` automatically.
  - all other source options are passed via `gst_source_props_str` (or full
    `gst_source_str`).
- Recommended workflow: confirm supported properties on your target with:
  - `gst-inspect-1.0 nvarguscamerasrc`
- `gCapture`/`nanoCapture` already do best-effort property checks internally for
  selected options; custom `gst_source_props_str` is passed through as-is.
