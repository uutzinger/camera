### OpenCV2 supported config keys

Core capture/output:
- `camera_res: (w, h)`            Capture size
- `output_res: (w, h)`            Optional CPU resize target
- `fps: float|int`                Requested framerate
- `buffersize: int`               FrameBuffer capacity (default 32)
- `buffer_overwrite: bool`        Overwrite oldest frames when full

Formats:
- `fourcc: str|int`               FOURCC request (e.g. `MJPG`, `YUY2`, `YU12`)
- `format: str`                   Raw format name (e.g. `YUYV`, `YUV420`, `SRGGB8`)
- `convert_rgb: bool`             If True (default), OpenCV backend returns BGR
- `convert_format: str`           Target format (default `BGR888`)

Controls:
- `exposure: float`
- `autoexposure: int`             -1 unchanged, 0 manual, 1 auto
- `gain: float`
- `wb_temp: int`
- `autowb: int`

Low-level/advanced:
- `buffer_count: int`             OpenCV CAP_PROP_BUFFERSIZE
- `flip: int`                     0..7 (same enum as cv2Capture)
- `test_pattern: bool|str`        Synthetic frame source for testing

