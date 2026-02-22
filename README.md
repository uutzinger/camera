# Camera Util

- [Camera Util](#camera-util)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Capture modules](#capture-modules)
  - [Camera Configuration](#camera-configuration)
  - [Example Programs](#example-programs)
  - [Changes](#changes)

## Overview

<img src="assetts/layout.svg" width="600" style="display: block;" alt="Camera Util" />

A collection of python threaded camera support routines for  

- USB and laptop internal webcams
- MIPI CSI cameras (Raspberry Pi, Jetson Nano)
- Teledyne/FLIR Spinnaker (USB)
- Basler pylon (best effort)
- RTP/RTSP streams

Support to save streams as  

- hdf5  
- tiff  
- avi
- mkv  

Support to process streams:

- background and flat field correction
- temporal highpass and lowpass filtering
- binning

The image acquisition runs in a background thread to achieve maximal frame rate and minimal latency.
 
Implementations are made to run on Windows, MacOS and Unix where possible.

## Requirements

- `OpenCV` for USB, CSI cameras
  - Windows uses cv2.CAP_MSMF
  - Darwin uses cv2.CAP_AVFOUNDATION
  - Linux uses cv2.CAP_V4L2
  - Jetson Nano uses cv2.CAP_GSTREAMER
  - RTSP typically uses FFmpeg backend in OpenCV wheels
- `GStreamer`
  - Jetson cameras
  - RTSP network streams
- `PySpin` and `Spinnacker` for FLIR/Teledyne cameras
- `pypylon` for Basler cameras
- `imageio-ffmpeg` for RTP receive on Windows/macOS
- `PiCamera2` for Raspberry Pi
- `tifffile` and `h5py`
- `numba` for processor acceleration

## Installation

### This Package <!-- omit from toc -->

If you cloned the repository:
- ``` cd "folder where you have this Readme.md file" ```
- ``` pip install -e . ``` or  ```python setup.py bdist_wheel``` and ```pip3 install .\dist\*.whl```

Directly from pypi:
- ```pip install camera-util```

Depending on the modules you need:

- [Support packages detailed installation instructions.](./Installation.md)

## Documentation

- [Capture Modules](Capture.md)
- [Streamer Modules](Streamer.md)
- [Processing Modules](Processing.md)
- [Camera Configuration](CameraConfig.md)
- [RTSP/RTP + GStreamer notes](RTSP_RTP_gstreamer.md)

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

## Capture modules

All modules have a Python threaded capture wrapper.

Frames are written into a single-producer/single-consumer ring buffer [`camera.buffer`](camera/capture/framebuffer.py).

Consumers pull images with: 
```
if camera.buffer.avail() > 0: 
   frame, ts_ms = camera.buffer.pull(copy=False)
```
`ts_ms` is a float timestamp in milliseconds.

For Qt application the consumer should pull images via a `QTimer`. The Qt wrapper does not emit a per-frame signal.
 
Documentation for [Capture Modules](Capture.md)

## Camera Configuration

For initialization you can provide a configuration dictionary.

### Specifications of your camera  <!-- omit from toc -->

On Windows, the Windows Camera Utility will give you resolution options and frames per second available on your camera. 

To investigate other options you can use OBS studio (or any other capture program), establish camera capture device and inspect video options.

When using OpenCV as camera interface `python3 list_cv2CameraProperties.py` will show all camera options the video system offers. When an option states `-1` it likely is not available for that camera. The program is located in [examples/list_cv2CameraProperties.py](examples/list_cv2CameraProperties.py).

For PiCamera2 we can use [examples/list_piCamera2Properties.py](examples/list_piCamera2Properties.py) to list supported formats.

For Basler pypylon cameras we can use [examples/list_pylonCameraProperties.py](examples/list_pylonCameraProperties.py) to inspect controls and trigger options.

Documentation for [Camera Config](CameraConfig.md)

## Example Programs

Try ```cv2_capture_display.py``` from ```./examples```.
You need to set the proper config file in the program. You should not need to edit python files in capture or streamer folder.

**Naming convention**: examples use backend-explicit names like ```cv2_*```, ```gcapture_*``` / ```qt_gcapture_*```, ```picamera2_*``` / ```qt_picamera2_*```, ```spin_capture_*```, and ```pylon_*``` / ```qt_pylon_*```.

**Capture + display**:

- ```cv2_capture_display.py``` OpenCV capture + display.
- ```qt_cv2_capture_display.py``` OpenCV + Qt display (polling via `QTimer`).
- ```gcapture_display.py``` GStreamer appsink capture (`gCapture`) + display.
- ```qt_gcapture_display.py``` GStreamer + Qt display (polling via `QTimer`).
- ```picamera2_capture_display.py``` Raspberry Pi Picamera2 capture + display (MAIN stream).
- ```qt_picamera2_capture_display.py``` Raspberry Pi Picamera2 + Qt display (polling via `QTimer`).
- ```spin_capture_display.py``` Spinnaker capture + display + FPS reporting.
- ```qt_spin_capture_display.py``` Spinnaker + Qt display (polling via `QTimer`).
- ```pylon_capture_display.py``` Basler pylon capture + display + FPS reporting.
- ```qt_pylon_capture_display.py``` Basler pylon + Qt display (polling via `QTimer`).
- ```rtsp_display.py``` RTSP receive + display.
- ```rtp_display.py``` RTP receive + display.

**Streaming (send RTP)**:

- ```cv2_capture_display_send2rtp.py``` OpenCV capture + display + send RTP.
- ```cv2_capture_display_send2rtp_process.py``` like above, but uses processes.
- ```gcapture_display_send2rtp.py``` gCapture + display + send RTP.

**Recording (HDF5 / TIFF / video)**:

- ```cv2_capture_savehdf5_display.py``` OpenCV capture + display + store to HDF5.
- ```cv2_capture_proc_savehdf5_display.py``` OpenCV capture + simple processing + display + HDF5.
- ```cv2_capture_savemkv_display.py``` OpenCV capture + display + store to MKV.
- ```spin_capture_savehdf5.py``` Spinnaker capture + store to HDF5.
- ```spin_capture_savehdf5_display.py``` Spinnaker capture + display + store to HDF5.
- ```spin_capture_proc_savehdf5_display.py``` Spinnaker capture + processing + display + store.
- ```spin_capture_savetiff_display.py``` Spinnaker capture + display + store to TIFF.
- ```pylon_capture_savehdf5_display.py``` Basler pylon capture + display + store to HDF5.

**Benchmarks / tests** (in `./testing`):

- ```benchmark_binning.py``` benchmark square/rectangular binning kernels.
- ```benchmark_runningsumprocessor.py``` benchmark runningsum vs one-pole processors.
- ```test_display.py``` display framerate test (no camera).
- ```test_circularbuffer.py``` ring buffer throughput/behavior test.
- ```test_savehdf5.py``` disk throughput test for HDF5 (no camera).
- ```test_savetiff.py``` disk throughput test for TIFF (no camera).
- ```test_saveavi.py``` disk throughput test for AVI (no camera; 3 color planes per image).
- ```test_savemkv.py``` disk throughput test for MKV/MP4V (no camera; 3 color planes per image).
- ```test_spin.py``` Spinnaker capture FPS test (no display).

## Example Camera Performance  <!-- omit from toc -->

### Sony IMX287 FLIR Spinnaker S BFS-U3-04S2M <!-- omit from toc -->

- 720x540 524fps
- auto_exposure off

### OV5647 OmniVision RasPi <!-- omit from toc -->

- auto_exposure 0: auto, 1:manual
- exposure in microseconds
- Max Resolution 2592x1944
- YU12, (YUYV, RGB3, JPEG, H264, YVYU, VYUY, UYVY, NV12, BGR3, YV12, NV21, BGR4)
- 320x240 90fps
- 640x480 90fps
- 1280x720 60fps
- 1920x1080 6.4fps
- 2592x1944 6.4fps

### IMX219 Sony RasPi <!-- omit from toc -->

- auto_exposure 0: auto, 1:manual
- exposure in microseconds
- Max Resolution 3280x2464
- YU12, (YUYV, RGB3, JPEG, H264, YVYU, VYUY, UYVY, NV12, BGR4)
- 320x240 90fps
- 640x480 90fps
- 1280x720 60fps
- 1920x1080 4.4fps
- 3280x2464 2.8fps

### ELP USB Camera RasPi <!-- omit from toc -->

- MJPG
- 320x240 and 640/480, 120fps
- auto_exposure, can not figure in MJPG mode
- auto_exposure = 0 -> static exposure
- exposure is about (exposure value / 10) in ms
- WB_TEMP 6500

### Dell Internal USB <!-- omit from toc -->

- 320x240, 30fps
- YUY2
- autoexposure ? 0.25, 0.74 -1. 0
- WB_TEMP 4600
- 1280x720, 30fps
- 620x480, 30fps
- 960x540, 30fps

## Changes

```text
2026 - Codereview, standardization of wrappers, Basler Cameras, Qt compatible wrapper with examples
2025 - Codereview, gcapture module, GI and FFMPEG fallbacks and standardized naming.
2022 - February added libcamera capture 
2022 - January added queue as intialization option, updated cv2Capture
2021 - November moved queue into class
2021 - November added rtp server and client
2021 - November added mkvServer, wheel installation, cleanup
2021 - October added aviServer and multicamera example, PySpin trigger fix
2021 - September updated PySpin trigger out polarity setting  
2020 - Initial release  
```
