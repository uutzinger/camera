# Camera Util

- [Camera Util](#camera-util)
  - [Overview](#overview)
  - [Capture modules](#capture-modules)
  - [Streamer modules](#streamer-modules)
  - [Installation](#installation)
  - [How to create camera config files](#how-to-create-camera-config-files)
  - [Video Flip](#video-flip)
  - [Exposure and Autoexposure](#exposure-and-autoexposure)
  - [Example Programs](#example-programs)
  - [Changes](#changes)

## Overview

A collection of python threaded camera support routines for  

- USB and laptop internal webcams
- RTSP streams
- MIPI CSI cameras (Raspberry Pi, Jetson Nano)
- Teledyne/FLIR blackfly (USB)
- Basler (not implemented yet)

Also support to save as  

- hdf5  
- tiff  
- avi
- mkv  

Supported OS  

- Windows
- MacOS
- Unix

The routines use OpenCV, gstreamer, libcamera, PiCamera2 or PySpin to interface with the camera. The image acquisition runs in a background thread to achieve maximal frame rate and minimal latency.

## Capture modules

### **blackflyCapture** <!-- omit from toc -->

Simplifies the settings needed for the Blackfly camera. You still need PySpin and Spinnaker software installed. Supports trigger out during frame exposure and trigger in for frame start. Optimized settings to achieve full frame rate with S BFS-U3-04S2M.

- Examples: [examples/blackfly_capture_display.py](examples/blackfly_capture_display.py), [examples/blackfly_capture_savehdf5_display.py](examples/blackfly_capture_savehdf5_display.py)

### **nanoCapture** <!-- omit from toc -->

gstreamer pipeline for nvidia conversion and nvarguscamera capture. Settings optimized for Sony IMX219 Raspi v2 Module.

- Examples (auto-selected on Jetson where applicable): [examples/cv2_capture_display.py](examples/cv2_capture_display.py), [examples/cv2_capture_display_send2rtp.py](examples/cv2_capture_display_send2rtp.py)

### **cv2Capture** <!-- omit from toc -->

cv2 capture architecture. The video subsystem is chosen based on the operating system.  

- Examples: [examples/cv2_capture_display.py](examples/cv2_capture_display.py), [examples/cv2_capture_savemkv_display.py](examples/cv2_capture_savemkv_display.py)

### **rtspCapture** <!-- omit from toc -->

RTSP capture architecture. RTSP camera can stream to multiple clients.

- If Python GI is available: uses GStreamer appsink (low latency, configurable).
- Otherwise (default on Windows/macOS with pip-installed OpenCV): uses OpenCV `VideoCapture(rtsp_url)` (typically FFmpeg backend).

- Examples: [examples/rtsp_display.py](examples/rtsp_display.py)

### **rtpCapture** <!-- omit from toc -->

RTP (UDP) stream is usually single-client.

- If Python GI is available: uses GStreamer appsink.
- Otherwise: uses FFmpeg via `imageio-ffmpeg` and requires an SDP file (`rtp_sdp`) to describe the RTP payload.

- Examples: [examples/rtp_display.py](examples/rtp_display.py)

### **piCapture** <!-- omit from toc -->

Interface for legacy picamera module.

- Example configs: [examples/configs/all_capture_configs.py](examples/configs/all_capture_configs.py)

### **piCamera2Capture** <!-- omit from toc -->

Interface for PiCamera2 module.

- Examples: [examples/raspi_capture_display.py](examples/raspi_capture_display.py)

### **gCapture** <!-- omit from toc -->

GStreamer appsink-based capture. Useful when you need a custom GStreamer pipeline (e.g., hardware decode/convert, CSI sources, RTSP).

- Examples: [examples/gcapture_display.py](examples/gcapture_display.py), [examples/gcapture_display_send2rtp.py](examples/gcapture_display_send2rtp.py)

## Streamer modules

### **rtpServer** <!-- omit from toc -->

Sends frames as an RTP/UDP H264 stream.

- Examples: [examples/cv2_capture_display_send2rtp.py](examples/cv2_capture_display_send2rtp.py), [examples/gcapture_display_send2rtp.py](examples/gcapture_display_send2rtp.py)
- Receiver example: [examples/rtp_display.py](examples/rtp_display.py)
- SDP template (for FFmpeg fallback): [examples/rtp_h264_pt96.sdp](examples/rtp_h264_pt96.sdp)

### **rtspServer** <!-- omit from toc -->

RTSP server for streaming frames to multiple clients.

- Server example: [examples/rtsp_server.py](examples/rtsp_server.py)
- Receiver example: [examples/rtsp_display.py](examples/rtsp_display.py)

### Storage servers <!-- omit from toc -->

Background writer threads for saving frames/cubes to disk.

- AVI (MJPG): [camera/streamer/avistorageserver.py](camera/streamer/avistorageserver.py)
- MKV (mp4v): [camera/streamer/mkvstorageserver.py](camera/streamer/mkvstorageserver.py)
- HDF5 arrays: [camera/streamer/h5storageserver.py](camera/streamer/h5storageserver.py)
- TIFF stacks: [camera/streamer/tiffstorageserver.py](camera/streamer/tiffstorageserver.py)


## Installation

### Requirements <!-- omit from toc -->

- `PySpin` for FLIR cameras
- `OpenCV` for USB, CSI cameras
  - Windows uses cv2.CAP_MSMF
  - Darwin uses cv2.CAP_AVFOUNDATION
  - Linux uses cv2.CAP_V4L2
  - Jetson Nano uses cv2.CAP_GSTREAMER
  - RTSP typically uses FFmpeg backend in OpenCV wheels
- `GStreamer`
  - RTSP network streams
- `imageio-ffmpeg` for RTP receive on Windows/macOS

### This Package <!-- omit from toc -->

- ``` cd "folder where you have this Readme.md file" ```
- ``` pip install . ``` or  ```python setup.py bdist_wheel``` and ```pip3 install .\dist\*.whl```

### GStreamer <!-- omit from toc -->

Used for RTSP/RTP pipelines and the `gCapture` appsink backend.

*Linux / Ubuntu*:

```bash
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-rtsp gstreamer1.0-alsa gstreamer1.0-pulseaudio v4l-utils
```

Linux (Python bindings for direct GStreamer use, needed by `gCapture`):

```bash
sudo apt install python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0
```

Verify Python GI works:

```bash
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
```

*Windows (GStreamer runtime)*:

Install GStreamer from [GStreamer](https://gstreamer.freedesktop.org/data/pkg/windows/) (typically the MSVC x86_64 installer). Add the GStreamer `bin` folder (e.g. `C:\gstreamer\1.0\msvc_x86_64\bin` or `C:\gstreamer\1.0\x86_64\bin`, depending on installer) to your `PATH`.

Verify GStreamer works:

```powershell
gst-inspect-1.0 --version
gst-device-monitor-1.0 Video/Source
gst-inspect-1.0 ksvideosrc
gst-inspect-1.0 dshowvideosrc
```

`gCapture` (and the GI backend of `rtpCapture`/`rtspCapture`) depends on Python GI bindings (`import gi`) which is not directly supported by default on Windows. Options:

- MSYS2 Python + PyGObject + GStreamer (then run your scripts from that MSYS2 environment)
- Conda / Miniconda with conda-forge packages (example):

```powershell
conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly
```

If you do not want MSYS2/Conda on Windows:

- Use `rtspCapture` via OpenCV (FFmpeg backend) for RTSP cameras.
- For raw RTP/UDP (`rtpCapture`), use the FFmpeg fallback and provide `rtp_sdp`.

### OpenCV <!-- omit from toc -->

For many cameras.

Unix: ```sudo apt install libopencv-dev python3-opencv``` for system level installation or ```pip3 install opencv-contrib-python``` for latest pip version or use instructions on [Qengineering (Ubuntu/Jetson)](https://qengineering.eu/install-opencv-on-jetson-nano.html) and/or [Qengineering (Raspberry Pi)](https://qengineering.eu/install-opencv-on-raspberry-pi.html) to compile your own. For Ubuntu you can also use my install script: [configure_opencv.sh](https://github.com/uutzinger/Linux_InstallScripts/blob/main/configure_opencv.sh)

Windows: ```pip3 install opencv-contrib-python```

### FFmpeg (RTP fallback) <!-- omit from toc -->

If you want `rtpCapture` to work on Windows/macOS without Python GI/GStreamer, install the bundled FFmpeg executable:

```bash
pip install imageio-ffmpeg
```

- The RTP fallback uses FFmpeg and requires an SDP file describing the RTP stream (set `rtp_sdp` in the capture config).
- The sender in `camera/streamer/rtpserver.py` can generate a matching SDP file when it uses the FFmpeg backend (pass `sdp_path=...`).
- RTSP does not need this; `rtspCapture` can usually use OpenCV+FFmpeg directly on an `rtsp://...` URL.

### Picamera2 <!-- omit from toc -->

Raspberry Pi: use `piCamera2Capture` (Picamera2/libcamera).

```bash
sudo apt install -y python3-picamera2
```

### Blackfly <!-- omit from toc -->

For [Teledyne BlackFly](https://www.teledynevisionsolutions.com/categories/cameras/) obtain [Spinnaker SDK and PySpin](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads)

Extract the files and on *Windows*: run the exe file. In the *Unix* shell: ```.\install_spinnaker.sh``` and ```pip install spinnaker_python...```

Please note, this usually requires an old version of Python and will not run with latest Python.

### TIFF and HDF5 <!-- omit from toc -->

For recording and storing image data.

On Unix ```sudo apt install libhdf5-dev libtiff-dev```

All systems: ```pip install tifffile imagecodecs h5py```

To get better tiff performance on Windows, [building libtiff is advised](https://github.com/uutzinger/Windows_Install_Scripts/blob/master/Buildinglibtiff.md)

## How to create camera config files

### Obtain specifications of your camera  <!-- omit from toc -->

On Windows, the Windows Camera Utility will give you resolution options and frames per second. To investigate other options you can use OBS studio (or any other capture program), establish camera capture device and inspect video options.

When using OpenCV as camera interface `python3 list_cv2CameraProperties.py` will show all camera options the video system offers. When an option states `-1` it likely is not available for that camera. The program is located in [camera/examples/list_cv2CameraProperties.py]([camera/examples/list_cv2CameraProperties.py)

### Create Configuration file  <!-- omit from toc -->

Use one of the existing camera configurations in [examples/configs/all_capture_configs.py](examples/configs/all_capture_configs.py)
 or create your own. 1) set appropriate resolution and frames per second. 2) figure out the exposure and autoexposure settings.

## Video Flip

| method | name                 | description                                         |
| ------ | -------------------- | --------------------------------------------------- |
| 0      | none                 | Identity (no rotation)                              |
| 1      | clockwise            | Rotate clockwise 90 degrees                         |
| 2      | rotate-180           | Rotate 180 degrees                                  |
| 3      | counterclockwise     | Rotate counter-clockwise 90 degrees                 |
| 4      | horizontal-flip      | Flip horizontally                                   |
| 5      | vertical-flip        | Flip vertically                                     |
| 6      | upper-left-diagonal  | Flip across upper left/lower right diagonal         |
| 7      | upper-right-diagonal | Flip across upper right/lower left diagonal         |
| 8      | automatic            | Select flip method based on image-orientation tag   |

## Exposure and Autoexposure

The config keys `exposure` and `autoexposure` are interpreted differently depending on the capture backend (OpenCV, GStreamer, Picamera2/libcamera, legacy picamera).

|key|what it does|typical values|
|---|---|---|
|`exposure`|Manual exposure request.|`>0`: time/value (often µs on PiCamera2/libcamera and Jetson nvargus). Some OpenCV webcam drivers use negative “stops-like” values (e.g. `-5` ≈ $2^{-5} = (1/32)$ on some devices; not portable). `0`/`-1`: often means “auto/leave default”, depending on backend.|
|`autoexposure`|Auto-exposure (AE) on/off request.|OpenCV: `-1` keep, `0` manual, `1` auto; many V4L2 cams also accept `0.25` (manual) / `0.75` (auto). Other backends either ignore this key or map it differently.|

Backend details:

- OpenCV (`cv2Capture`):
  - `autoexposure = -1` leaves AE unchanged; `0` requests manual AE mode; `1` requests auto AE mode.
  - Some Linux/V4L2 drivers commonly accept `0.25` (manual) and `0.75` (auto). Windows/macOS backends can differ.
  - `exposure > 0` requests manual exposure and the code attempts to disable AE first. The numeric meaning of `CAP_PROP_EXPOSURE` is driver-specific (some cameras use negative values, others use milliseconds/microseconds-like scales).
  - To inspect what your OpenCV backend/camera supports, run `python3 examples/list_cv2CameraProperties.py`.

- GStreamer (`gCapture`):
  - `gCapture` builds a pipeline and the meaning of “exposure” depends on the selected source element:
    - Jetson `nvarguscamerasrc`: `exposure` is treated as microseconds (converted internally) and applied via `exposuretimerange` when supported.
    - `libcamerasrc`: exposure/AE are controlled via libcamera controls, but the exact property interface is element-version specific. Use `gst-inspect-1.0 libcamerasrc` and set the element’s control-related properties via `gst_source_props_str`, or provide a complete custom source with `gst_source_str`.
    - `v4l2src`: exposure is typically controlled via V4L2 controls (recommended: `v4l2-ctl` from `v4l-utils`). Some GStreamer builds expose an `extra-controls`-style property on `v4l2src`, but it is not consistent across platforms/drivers.
    - Windows `ksvideosrc` / `dshowvideosrc`: many cameras do not expose manual exposure controls through these elements. If the element exposes a property for it, you can set it via `gst_source_props_str`; otherwise use vendor tools / OS camera settings (or configure exposure through a separate API).
  - Use `gst-inspect-1.0 <element>` to discover supported properties and set them via `gst_source_props_str` or a fully custom `gst_source_str`.

- FLIR/Teledyne Blackfly (`blackflyCapture`, PySpin):
  - `exposure` is in microseconds and maps to `ExposureTime` (requires `ExposureAuto` off and `ExposureMode=Timed`).
  - `autoexposure`: `0` off, `1` on (continuous/once depending on camera settings).

- Raspberry Pi Picamera2/libcamera (`piCamera2Capture`):
  - `exposure` is interpreted as microseconds and mapped to libcamera `ExposureTime`.
  - `autoexposure` maps to libcamera `AeEnable` (`>0` enables, `0` disables, `-1` leaves unchanged).

- Raspberry Pi legacy picamera (`piCapture`):
  - Exposure is controlled via `shutter_speed` in microseconds.
  - In this backend, `exposure <= 0` uses automatic exposure (`exposure_mode='auto'`); `exposure > 0` requests manual exposure (`exposure_mode='off'` plus `shutter_speed`).

Maximum usable exposure time is typically limited by frame interval. Roughly, max exposure is about $\frac{10^6}{\mathrm{fps}}$ microseconds (camera/driver may clamp differently).

## Example Programs

Try ```cv2_capture_display.py``` from ```./examples```.
You need to set the proper config file in the program. You should not need to edit python files in capture or streamer folder.

**Naming convention**: examples use backend-explicit names like ```cv2_*```, ```gcapture_*```, ```raspi_*```, and ```blackfly_capture_*```.

**Capture + display**:

- ```cv2_capture_display.py``` OpenCV capture + display.
- ```gcapture_display.py``` GStreamer appsink capture (`gCapture`) + display.
- ```raspi_capture_display.py``` Raspberry Pi capture + display (prefers Picamera2, falls back to OpenCV).
- ```blackfly_capture_display.py``` Blackfly capture + display + FPS reporting.
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
- ```blackfly_capture_savehdf5.py``` Blackfly capture + store to HDF5.
- ```blackfly_capture_savehdf5_display.py``` Blackfly capture + display + store to HDF5.
- ```blackfly_capture_proc_savehdf5_display.py``` Blackfly capture + processing + display + store.
- ```blackfly_capture_savetiff_display.py``` Blackfly capture + display + store to TIFF.

**Benchmarks / tests**:

- ```test_display.py``` display framerate test (no camera).
- ```test_savehdf5.py``` disk throughput test for HDF5 (no camera).
- ```test_savetiff.py``` disk throughput test for TIFF (no camera).
- ```test_saveavi.py``` disk throughput test for AVI (no camera; 3 color planes per image).
- ```test_savemkv.py``` disk throughput test for MKV/MP4V (no camera; 3 color planes per image).
- ```test_blackfly.py``` Blackfly capture FPS test (no display).

## Example Camera Performance  <!-- omit from toc -->

### Sony IMX287 FLIR Blackfly S BFS-U3-04S2M <!-- omit from toc -->

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
- auto_exposure, can not figure out out in MJPG mode
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

## Pip upload  <!-- omit from toc -->

```bash
python3 -m pip install --upgrade pip build twine
rm -rf dist build *.egg-info
python3 -m build
python3 -m twine check dist/*
python3 -m pip install dist/*.whl
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-pypi-token>
python -m twine upload dist/*
```

## Changes

```text
2025 - Codereview, gcapture module, GI and FFMPEG fallbacks and standardized naming.
2022 - February added libcamera capture for Raspian Bullseye
2022 - January added queue as intialization option, updated cv2Capture
2021 - November moved queue into class
2021 - November added rtp server and client
2021 - November added mkvServer, wheel installation, cleanup
2021 - October added aviServer and multicamera example, PySpin trigger fix
2021 - September updated PySpin trigger out polarity setting  
2020 - Initial release  
```
