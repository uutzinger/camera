## Streamer Modules

Threaded processors live in `camera/streamer/`.

### **rtpServer** <!-- omit from toc -->

Sends frames as an RTP/UDP H264 stream.

- Design notes: [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md)

- Examples: 
  - [examples/cv2_capture_display_send2rtp.py](examples/cv2_capture_display_send2rtp.py)
  - [examples/gcapture_display_send2rtp.py](examples/gcapture_display_send2rtp.py)
- Receiver example: 
  - [examples/rtp_display.py](examples/rtp_display.py)
- SDP template (for FFmpeg fallback): 
  - [examples/rtp_h264_pt96.sdp](examples/rtp_h264_pt96.sdp)

### **rtspServer** <!-- omit from toc -->

RTSP server for streaming frames to multiple clients.

- Design notes: [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md)

- Queue input accepts `frame` or `(ts_ms, frame)`.
- Timestamp handling can be set with `timestamp_unit` (`auto`, `ms`, `s`, `ns`); wrapper/capture timestamps are `ts_ms`.
- Server example: 
  - [examples/rtsp_server.py](examples/rtsp_server.py)
- Receiver example: 
  - [examples/rtsp_display.py](examples/rtsp_display.py)

### Storage servers <!-- omit from toc -->

Background writer threads for saving frames/cubes to disk.

- AVI (MJPG): [camera/streamer/avistorageserver.py](camera/streamer/avistorageserver.py)
- MKV (mp4v): [camera/streamer/mkvstorageserver.py](camera/streamer/mkvstorageserver.py)
- HDF5 arrays: [camera/streamer/h5storageserver.py](camera/streamer/h5storageserver.py)
- TIFF stacks: [camera/streamer/tiffstorageserver.py](camera/streamer/tiffstorageserver.py)

### Streamer tests <!-- omit from toc -->

- HDF5 writer throughput: [testing/test_savehdf5.py](testing/test_savehdf5.py)
- TIFF writer throughput: [testing/test_savetiff.py](testing/test_savetiff.py)
- AVI writer throughput: [testing/test_saveavi.py](testing/test_saveavi.py)
- MKV/MP4 writer throughput: [testing/test_savemkv.py](testing/test_savemkv.py)
