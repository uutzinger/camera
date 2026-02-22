## Capture

- [cv2Capture](#cv2capture)
- [piCamera2Capture](#picamera2capture)
- [spinCapture](#spincapture)
- [pylonCapture](#pyloncapture)
- [nanoCapture](#nanocapture)
- [rtspCapture](#rtspcapture)
- [rtpCapture](#rtpcapture)
- [piCapture](#picapture)
- [gCapture](#gcapture)

### Capture tests <!-- omit from toc -->

- Generic display timing test: [testing/test_display.py](testing/test_display.py)
- Frame ring buffer test: [testing/test_circularbuffer.py](testing/test_circularbuffer.py)
- Spinnaker capture FPS test: [testing/test_spin.py](testing/test_spin.py)

## **cv2Capture** <!-- omit from toc -->

The video subsystem is chosen based on the operating system.  
  
  Threaded, non-Qt: [cv2capture.py](camera/capture/cv2capture.py) and Qt5/Qt6: [cv2captureQt.py](camera/capture/cv2captureQt.py)
- [Methods, Properties and Setters](OpenCV.md)
- [Config Keys](OpenCV_Config.md)
- Examples:
  - [examples/cv2_capture_display.py](examples/cv2_capture_display.py) 
  - [examples/qt_cv2_capture_display.py](examples/qt_cv2_capture_display.py), 
  - [examples/cv2_capture_savemkv_display.py](examples/cv2_capture_savemkv_display.py)

### **piCamera2Capture** <!-- omit from toc -->

Interface for Raspberry Pi Picamera2/libcamera 

Threaded, non-Qt: [picamera2capture.py](camera/capture/picamera2capture.py) and Qt5/Qt6: [picamera2captureQt.py](camera/capture/picamera2captureQt.py)
- [Methods, Properties and Setter](piCamera2.md)
- [Config Keys](piCamera2_Config.md)
- Examples:
  - Qt display (polls via `QTimer`): [examples/qt_picamera2_capture_display.py](examples/qt_picamera2_capture_display.py)
  - MAIN stream display: [examples/picamera2_capture_display.py](examples/picamera2_capture_display.py)
  - RAW / max-FPS / low-latency: [examples/picamera2_capture_raw_maxfps_lowlatency.py](examples/picamera2_capture_raw_maxfps_lowlatency.py)


### **spinCapture** <!-- omit from toc -->

Simplifies the settings needed for the Spinnaker camera. This module depends on PySpin and Spinnaker software. PySpin supports trigger out during frame exposure and trigger in for frame start.

 Optimized settings to achieve full frame rate with S BFS-U3-04S2M.

Threaded, non-Qt: [spincapture.py](camera/capture/spincapture.py) and Qt5/Qt6: [spincaptureQt.py](camera/capture/spincaptureQt.py)
- [Methods, Properties and Setter](spin.md)
- [Config Keys](spin_Config.md)

- Examples: 
  - Regular [examples/spin_capture_display.py](examples/spin_capture_display.py), 
  - With processing and saving: [examples/spin_capture_savehdf5_display.py](examples/spin_capture_savehdf5_display.py),
  - QT display (polls via QTimer) [examples/qt_spin_capture_display.py](examples/qt_spin_capture_display.py)

### **pylonCapture** <!-- omit from toc -->

Basler pylon/pypylon wrapper with the same core + threaded + Qt architecture.
Control availability is camera-model dependent and unsupported controls are ignored with logs.

Threaded, non-Qt: [pyloncapture.py](camera/capture/pyloncapture.py) and Qt5/Qt6: [pyloncaptureQt.py](camera/capture/pyloncaptureQt.py)
- [Methods, Properties and Setter](pylon.md)
- [Config Keys](pylon_Config.md)

- Examples:
  - Regular [examples/pylon_capture_display.py](examples/pylon_capture_display.py)
  - With HDF5 saving: [examples/pylon_capture_savehdf5_display.py](examples/pylon_capture_savehdf5_display.py)
  - QT display (polls via QTimer) [examples/qt_pylon_capture_display.py](examples/qt_pylon_capture_display.py)
  - Property/capability listing: [examples/list_pylonCameraProperties.py](examples/list_pylonCameraProperties.py)

### **nanoCapture** <!-- omit from toc -->

Jetson-focused wrapper around `gCapture`.

- Default source is `nvarguscamerasrc` (`gst_source='nvargus'`).
- Keeps legacy `nanoCapture` import path/API for Jetson examples.
- [nvarguscamerasrc Options](gCapture_nvarguscam.md)
- Inherits gCapture backend behavior:
  - Preferred: GI/PyGObject + appsink
  - Fallback: OpenCV `CAP_GSTREAMER` if GI is unavailable or pipeline init fails

- Examples: 
  - [examples/cv2_capture_display.py](examples/cv2_capture_display.py), 
  - [examples/cv2_capture_display_send2rtp.py](examples/cv2_capture_display_send2rtp.py)


### **rtspCapture** <!-- omit from toc -->

RTSP network server capture. An RTSP camera can stream to multiple clients.

Threaded, non-Qt: [rtspcapture.py](camera/capture/rtspcapture.py) and Qt5/Qt6: [rtspcaptureQt.py](camera/capture/rtspcaptureQt.py)
- [Methods, Properties and Setter](rtsp.md)
- [Config Keys](rtsp_Config.md)

- Backends:
  - If Python GI is available: uses GStreamer appsink (low latency, configurable).
  - Otherwise uses OpenCV `VideoCapture(rtsp_url)` (typically through FFmpeg backend) (default on Windows/macOS with pip-installed OpenCV).
  - Linux fallback: OpenCV + CAP_GSTREAMER pipeline (useful on Jetson builds).

- Examples: 
- [examples/rtsp_display.py](examples/rtsp_display.py)
  - [examples/qt_rtsp_capture_display.py](examples/qt_rtsp_capture_display.py)

### **rtpCapture** <!-- omit from toc -->

RTP (UDP) network stream is usually single-client.

threaded, non-Qt: [rtpcapture.py](camera/capture/rtpcapture.py) and Qt5/Qt6: [rtpcaptureQt.py](camera/capture/rtpcaptureQt.py)
- [Methods, Properties and Setter](rtp.md)
- [Config Keys](rtp_Config.md)

- Backends:
  - If Python GI is available: uses GStreamer appsink.
  - Otherwise: uses FFmpeg via `imageio-ffmpeg` and requires an SDP file (`rtp_sdp`) to describe the RTP payload.

- Examples: 
  - [examples/rtp_display.py](examples/rtp_display.py)
  - [examples/qt_rtp_capture_display.py](examples/qt_rtp_capture_display.py)

### **piCapture** <!-- omit from toc -->

Interface for legacy picamera software module. This is now replaced with picamera2 on latest raspian.

Need to fix

- Example: 
  - [examples/configs/all_capture_configs.py](examples/configs/all_capture_configs.py)


### **gCapture** <!-- omit from toc -->

Generic GStreamer capture with source auto-detection and custom source options.

Threaded, non-Qt: [gcapture.py](camera/capture/gcapture.py) and Qt5/Qt6: [gcaptureQt.py](camera/capture/gcaptureQt.py)
- [Methods, Properties and Setter](gCapture.md)
- [Config Keys](gCapture_Config.md)
- Source selection via `gst_source` / `source` or full custom `gst_source_str`.
- Suitable for libcamera, nvargus, v4l2, and Windows GStreamer webcam sources.
- Source option references:
  - [nvarguscamerasrc](gCapture_nvarguscam.md)
  - [libcamerasrc](gCapture_libcameracam.md)
  - [v4l2src](gCapture_v4l2cam.md)
  - [ksvideosrc](gCapture_ksvideocam.md)
  - [dshowvideosrc](gCapture_dshowcam.md)
- Backends:
  - Preferred: GI/PyGObject + appsink
  - Fallback: OpenCV `CAP_GSTREAMER` if GI is unavailable or pipeline init fails

- Examples: 
  - [examples/gcapture_display.py](examples/gcapture_display.py)
  - [examples/qt_gcapture_display.py](examples/qt_gcapture_display.py)
  - [examples/gcapture_display_send2rtp.py](examples/gcapture_display_send2rtp.py)
