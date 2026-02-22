### RTSP supported config keys

Core capture/output:
- `rtsp: str`                     RTSP URL
- `output_res: (w, h)`            Optional CPU resize target
- `buffersize: int`               FrameBuffer capacity (default 32)
- `buffer_overwrite: bool`        Overwrite oldest frames when full

Backend selection:
- `prefer_gi: bool`               Prefer GI/GStreamer appsink when available
- `gpu: bool`                     Prefer GPU decode elements on non-Linux
- `rtsp_latency: int`             GStreamer rtspsrc latency (ms, default 10)

Transforms:
- `flip: int`                     0..7 (same enum as cv2Capture)

### Notes
- Stream is opened in `__init__` with the provided configs. `start()/stop()` only
  control frame capture, not stream open/close.
