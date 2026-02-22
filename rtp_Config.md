### RTP supported config keys

Core capture/output:
- `port: int`                     UDP port for RTP stream (default 554)
- `output_res: (w, h)`            Optional CPU resize target
- `buffersize: int`               FrameBuffer capacity (default 32)
- `buffer_overwrite: bool`        Overwrite oldest frames when full

RTP payload:
- `rtp_encoding_name: str`        RTP encoding name (default `H264`)
- `rtp_payload: int`              RTP payload type (default 96)
- `rtp_clock_rate: int`           RTP clock rate (default 90000)

Backend selection:
- `prefer_gi: bool`               Prefer GI/GStreamer appsink when available
- `gpu: bool`                     Prefer GPU decode elements on non-Linux

FFmpeg fallback (when GI/OpenCV pipeline is unavailable):
- `rtp_sdp: str` or `sdp: str`     SDP file describing the RTP stream
- `camera_res: (w, h)`            Required to size raw frame reads (or set `output_res`)

Transforms:
- `flip: int`                     0..7 (same enum as cv2Capture)

### Notes
- Stream is opened in `__init__` with the provided configs. `start()/stop()` only
  control frame capture, not stream open/close.
