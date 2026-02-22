# Documentation Index

Complete reference table for all camera-util documentation files organized by module type.

## Complete Documentation Reference

| Category | Module/Component | Methods & Properties | Configuration | Additional References |
|----------|------------------|---------------------|---------------|----------------------|
| **CAPTURE** | cv2Capture | [OpenCV.md](OpenCV.md) | [OpenCV_Config.md](OpenCV_Config.md) | |
| **CAPTURE** | piCamera2Capture | [piCamera2.md](piCamera2.md) | [piCamera2_Config.md](piCamera2_Config.md) | |
| **CAPTURE** | spinCapture | [spin.md](spin.md) | [spin_Config.md](spin_Config.md) | |
| **CAPTURE** | pylonCapture | [pylon.md](pylon.md) | [pylon_Config.md](pylon_Config.md) | |
| **CAPTURE** | rtspCapture | [rtsp.md](rtsp.md) | [rtsp_Config.md](rtsp_Config.md) | [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md) |
| **CAPTURE** | rtpCapture | [rtp.md](rtp.md) | [rtp_Config.md](rtp_Config.md) | [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md) |
| **CAPTURE** | gCapture | [gCapture.md](gCapture.md) | [gCapture_Config.md](gCapture_Config.md) | See gCapture sources below |
| **CAPTURE** | nanoCapture | [gCapture.md](gCapture.md) | [gCapture_Config.md](gCapture_Config.md) | [gCapture_nvarguscam.md](gCapture_nvarguscam.md) |
| **CAPTURE** | piCapture | *(legacy - needs fix)* | | |
| | | | | |
| **gCapture Sources** | nvarguscamerasrc | | | [gCapture_nvarguscam.md](gCapture_nvarguscam.md) |
| **gCapture Sources** | libcamerasrc | | | [gCapture_libcameracam.md](gCapture_libcameracam.md) |
| **gCapture Sources** | v4l2src | | | [gCapture_v4l2cam.md](gCapture_v4l2cam.md) |
| **gCapture Sources** | ksvideosrc | | | [gCapture_ksvideocam.md](gCapture_ksvideocam.md) |
| **gCapture Sources** | dshowvideosrc | | | [gCapture_dshowcam.md](gCapture_dshowcam.md) |
| | | | | |
| **STREAMER** | rtpServer | [Streamer.md](Streamer.md) | | [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md) |
| **STREAMER** | rtspServer | [Streamer.md](Streamer.md) | | [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md) |
| **STREAMER** | aviStorageServer | [Streamer.md](Streamer.md) | | |
| **STREAMER** | mkvStorageServer | [Streamer.md](Streamer.md) | | |
| **STREAMER** | h5StorageServer | [Streamer.md](Streamer.md) | | |
| **STREAMER** | tiffStorageServer | [Streamer.md](Streamer.md) | | |
| | | | | |
| **PROCESSOR** | bgflatProcessor | [Processing.md](Processing.md) | | |
| **PROCESSOR** | highpassProcessor | [Processing.md](Processing.md) | | |
| **PROCESSOR** | lowpassProcessor | [Processing.md](Processing.md) | | |
| **PROCESSOR** | threebandProcessor | [Processing.md](Processing.md) | | |
| **PROCESSOR** | utilities (binning) | [Processing.md](Processing.md) | | |
| | | | | |
| **GENERAL** | Overview | [README.md](README.md) | | |
| **GENERAL** | Main Documentation | [Capture.md](Capture.md), [Streamer.md](Streamer.md), [Processing.md](Processing.md) | | |
| **GENERAL** | Installation | [Installation.md](Installation.md) | | |
| **GENERAL** | Camera Configuration | [CameraConfig.md](CameraConfig.md) | | |
| **GENERAL** | RTSP/RTP + GStreamer | [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md) | | |
| **GENERAL** | PyPI Upload | [PypiUpload.md](PypiUpload.md) | | |
| **GENERAL** | Wrapper Requirements | [WRAPPER_REQUIREMENTS.md](WRAPPER_REQUIREMENTS.md) | | |

## Quick Reference by Module

### Capture Modules
- **cv2Capture**: [OpenCV.md](OpenCV.md) • [OpenCV_Config.md](OpenCV_Config.md)
- **piCamera2Capture**: [piCamera2.md](piCamera2.md) • [piCamera2_Config.md](piCamera2_Config.md)
- **spinCapture**: [spin.md](spin.md) • [spin_Config.md](spin_Config.md)
- **pylonCapture**: [pylon.md](pylon.md) • [pylon_Config.md](pylon_Config.md)
- **rtspCapture**: [rtsp.md](rtsp.md) • [rtsp_Config.md](rtsp_Config.md) • [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md)
- **rtpCapture**: [rtp.md](rtp.md) • [rtp_Config.md](rtp_Config.md) • [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md)
- **gCapture**: [gCapture.md](gCapture.md) • [gCapture_Config.md](gCapture_Config.md)
  - Sources: [nvarguscam](gCapture_nvarguscam.md) • [libcamerasrc](gCapture_libcameracam.md) • [v4l2src](gCapture_v4l2cam.md) • [ksvideosrc](gCapture_ksvideocam.md) • [dshowvideosrc](gCapture_dshowcam.md)
- **nanoCapture**: Uses gCapture backend with [gCapture_nvarguscam.md](gCapture_nvarguscam.md)

### Streamer Modules
- **rtpServer**: [Streamer.md](Streamer.md) • [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md)
- **rtspServer**: [Streamer.md](Streamer.md) • [RTSP_RTP_gstreamer.md](RTSP_RTP_gstreamer.md)
- **Storage Servers** (AVI, MKV, HDF5, TIFF): [Streamer.md](Streamer.md)

### Processor Modules
All processors documented in [Processing.md](Processing.md):
- bgflatProcessor
- highpassProcessor
- lowpassProcessor
- threebandProcessor
- utilities (binning helpers)

## File Type Summary

| Documentation Type | Files |
|-------------------|-------|
| **Methods & Properties** | OpenCV.md, piCamera2.md, spin.md, pylon.md, rtsp.md, rtp.md, gCapture.md, Streamer.md, Processing.md |
| **Configuration Keys** | OpenCV_Config.md, piCamera2_Config.md, spin_Config.md, pylon_Config.md, rtsp_Config.md, rtp_Config.md, gCapture_Config.md |
| **GStreamer Sources** | gCapture_nvarguscam.md, gCapture_libcameracam.md, gCapture_v4l2cam.md, gCapture_ksvideocam.md, gCapture_dshowcam.md |
| **General/Setup** | README.md, Installation.md, CameraConfig.md, RTSP_RTP_gstreamer.md, PypiUpload.md, WRAPPER_REQUIREMENTS.md |
| **Module Overviews** | Capture.md, Streamer.md, Processing.md |

## Documentation Tree Structure

```
camera-util/
├── README.md (Main entry point)
├── Installation.md
├── CameraConfig.md
│
├── Capture.md (Overview of all capture modules)
│   ├── cv2Capture/
│   │   ├── OpenCV.md
│   │   └── OpenCV_Config.md
│   ├── piCamera2Capture/
│   │   ├── piCamera2.md
│   │   └── piCamera2_Config.md
│   ├── spinCapture/
│   │   ├── spin.md
│   │   └── spin_Config.md
│   ├── pylonCapture/
│   │   ├── pylon.md
│   │   └── pylon_Config.md
│   ├── rtspCapture/
│   │   ├── rtsp.md
│   │   ├── rtsp_Config.md
│   │   └── RTSP_RTP_gstreamer.md
│   ├── rtpCapture/
│   │   ├── rtp.md
│   │   ├── rtp_Config.md
│   │   └── RTSP_RTP_gstreamer.md
│   ├── gCapture/
│   │   ├── gCapture.md
│   │   ├── gCapture_Config.md
│   │   └── Sources/
│   │       ├── gCapture_nvarguscam.md
│   │       ├── gCapture_libcameracam.md
│   │       ├── gCapture_v4l2cam.md
│   │       ├── gCapture_ksvideocam.md
│   │       └── gCapture_dshowcam.md
│   └── nanoCapture/ → gCapture
│
├── Streamer.md (Overview of all streamer modules)
│   ├── rtpServer → RTSP_RTP_gstreamer.md
│   ├── rtspServer → RTSP_RTP_gstreamer.md
│   └── Storage servers (AVI, MKV, HDF5, TIFF)
│
└── Processing.md (All processor modules)
    ├── bgflatProcessor
    ├── highpassProcessor
    ├── lowpassProcessor
    ├── threebandProcessor
    └── utilities (binning)
```
