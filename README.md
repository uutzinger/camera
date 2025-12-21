# Camera Util

- [Camera Util](#camera-util)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [Camera](#camera)
    - [GStreamer](#gstreamer)
    - [OpenCV](#opencv)
    - [Picamera2](#picamera2)
    - [Blackfly](#blackfly)
    - [TIFF and HDF5](#tiff-and-hdf5)
  - [How to create camera config files](#how-to-create-camera-config-files)
  - [Run Example](#run-example)
    - [Example Programs](#example-programs)
  - [Capture modules](#capture-modules)
    - [**blackflyCapture**](#blackflycapture)
    - [**nanoCapture**](#nanocapture)
    - [**cv2Capture**](#cv2capture)
    - [**rtspCapture**](#rtspcapture)
    - [**piCapture**](#picapture)
    - [**piCamera2Capture**](#picamera2capture)
    - [**gCapture**](#gcapture)
  - [Changes](#changes)
  - [Example Camera Performance](#example-camera-performance)
    - [Sony IMX287 FLIR Blackfly S BFS-U3-04S2M](#sony-imx287-flir-blackfly-s-bfs-u3-04s2m)
    - [OV5647 OmniVision RasPi](#ov5647-omnivision-raspi)
    - [IMX219 Sony RasPi](#imx219-sony-raspi)
    - [ELP USB Camera RasPi](#elp-usb-camera-raspi)
    - [Dell Internal USB](#dell-internal-usb)
  - [Pip upload](#pip-upload)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Overview

A collection of python threaded camera support routines for  

- USB and laptop internal webcams
- RTSP streams
- MIPI CSI cameras (Raspberry Pi, Jetson Nano)
- FLIR blackfly (USB)

Also support to save as  

- HDF5  
- tiff  
- avi
- mkv  

Supported OS  

- Windows
- MacOS
- Unix

The routines use OpenCV, gstreamer or PySpin to interface with the camera.
The image acquisition runs in a background thread to achieve maximal frame rate and minimal latency.

## Requirements

- PySpin for FLIR cameras
- opencv for USB, CSI cameras
  - Windows uses cv2.CAP_MSMF
  - Darwin uses cv2.CAP_AVFOUNDATION
  - Linux uses cv2.CAP_V4L2
  - Jetson Nano uses cv2.CAP_GSTREAMER
  - RTSP uses cv2.CAP_GSTREAMER
- GStreamer
  - RTSP network streams

## Installation

### Camera

Required for this package.

- ``` cd "folder where you have this Readme.md file" ```
- ``` pip install . ``` or  ```python setup.py bdist_wheel``` and ```pip3 install .\dist\*.whl```

For a quick local install (development / examples):

- ```pip install -r requirements.txt```

### GStreamer

Used for RTSP/RTP pipelines and the `gCapture` appsink backend.

Linux / Ubuntu (no libcamera needed):

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

Windows: Install gstreamer from [GStreamer](https://gstreamer.freedesktop.org/data/pkg/windows/ ) and add `C:\gstreamer\1.0\x86_64\bin` to Path variable in Environment

### OpenCV

For regular system cameras.

Unix: ```sudo apt install libopencv-dev python3-opencv``` or use instructions on [Qengineering (Ubuntu/Jetson)](https://qengineering.eu/install-opencv-on-jetson-nano.html) and/or [Qengineering (Raspberry Pi)](https://qengineering.eu/install-opencv-on-raspberry-pi.html)

Windows: ```pip3 install opencv-contrib-python```

### Picamera2

Raspberry Pi: use `piCamera2Capture` (Picamera2/libcamera). This is platform-specific.

### Blackfly

For [Teledyne BlackFly](https://www.teledynevisionsolutions.com/categories/cameras/)  

Obtain Spinnaker SDK and PySpin from [Teledyne](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads)

Extract the files and on *Windows*: run the exe file. In the *Unix* shell: ```.\install_spinnaker.sh``` and ```pip install spinnaker_python...```

Please note, this usually requires an old version of Python.

### TIFF and HDF5

For recording and storing image data.

On Unix ```sudo apt install libhdf5-dev libtiff-dev```

and for all systems: ```pip install tifffile imagecodecs h5py```

To get better tiff performance on Windows, [building libtiff is advised](https://github.com/uutzinger/Windows_Install_Scripts/blob/master/Buildinglibtiff.md)


## How to create camera config files

A. Specifications of your camera  

On Windows, the Camera utility will give you resolution options and frames per second. To investigate other options you can use OBS studio (or any other capture program), establish camera capture device and inspect video options.

When using OpenCV as camera interface `python3 list_cv2CameraProperties.py` will show all camera options the video system offers. When an option states `-1` it likely is not available for that camera. The program is located in ```./examples```

B. Configuration file  

Use one of the existing camera configurations in ```examples/configs``` or create your own.
As first step set appropriate resolution and frames per second.
As second step figure out the exposure and autoexposure settings.

## Run Example

Run ```cv2_capture_display.py``` from ```./examples```.
You need to set the proper config file in the program. You should not need to edit python files in capture or streamer folder.

### Example Programs

**Display**: In general display should occur in main python program as it interfaces with user input and display system.

**Queue**: is used to transfer data between the main program and capture and storage threads.

**Naming convention**: examples use backend-explicit names like ```cv2_*```, ```gcapture_*```, ```raspi_*```, and ```blackfly_capture_*```.

**Examples**:

- ```cv2_capture_display.py``` tests OpenCV-based capture.
- ```gcapture_display.py``` tests gCapture (GStreamer appsink capture).
- ```raspi_capture_display.py``` prefers PiCamera2 on Raspberry Pi (falls back to OpenCV).
- ```blackfly_capture_display.py``` tests the Blackfly capture module, displays images and reports framerate.
- ```rtsp_display.py``` tests RTSP capture/display.
- ```rtp_display.py``` tests RTP receive/display.
- ```cv2_capture_display_send2rtp.py``` capture + display + send RTP.
- ```cv2_capture_display_send2rtp_process.py``` like above, but uses processes.
- ```gcapture_display_send2rtp.py``` gCapture + display + send RTP.


- ```cv2_capture_savehdf5_display.py``` display and store cubes to HDF5.
- ```cv2_capture_proc_savehdf5_display.py``` capture + simple processing + display + HDF5.
- ```cv2_capture_savemkv_display.py``` display and store to MKV.
- ```blackfly_capture_savehdf5.py``` Blackfly capture + store cubes to HDF5.
- ```blackfly_capture_savehdf5_display.py``` Blackfly capture + display + store cubes to HDF5.
- ```blackfly_capture_proc_savehdf5_display.py``` Blackfly capture + process + display + store cubes.
- ```blackfly_capture_savetiff_display.py``` Blackfly capture + display + store cubes to TIFF.


- ```test_display.py``` testing of opencv display framerate, no camera, just refresh rate.
- ```test_savehdf5.py``` testing of the disk throughput with HDF5, no camera
- ```test_savetiff.py``` testing of the disk throughput with tiff, no camera
- ```test_saveavi.py``` testing of the disk throughput with avi, no camera, only 3 color planes per image possible
- ```test_savemkv.py``` testing of the disk throughput with mkv/mp4v, no camera, only 3 color planes per image possible
- ```test_blackfly.py``` tests the blackfly capture module and reports framerate, no display

## Capture modules

### **blackflyCapture**

Simplifies the settings needed for the Blackfly camera.
Supports trigger out during frame exposure and trigger in for frame start. Optimized settings to achieve full frame rate with S BFS-U3-04S2M.

### **nanoCapture**

Uses gstreamer pipeline for Jetson Nano.  
Pipline for nvidia conversion and nvarguscamera capture.  
Settings optimized for Sony IMX219 Raspi v2 Module.

### **cv2Capture**

Uses the cv2 capture architecture.  
The video subsystem is choosen based on the operating system.  

### **rtspCapture**

gstreamer based rtsp network stream capture for all platforms.  
gstreamer is called through OpenCV.  
By default OpenCV supports ffmpeg and not gstreamer. Jetson Nano does not support ffmpeg but opencv is prebuilt with gstreamer for that platform.

### **piCapture**

Interface for legacy picamera module.

### **piCamera2Capture**

Interface for PiCamera2 module.

### **gCapture**

GStreamer appsink-based capture. Useful when you need a custom GStreamer pipeline (e.g., hardware decode/convert, CSI sources, RTSP).

## Changes

```text
2025 - Codereview, gcapture module and standardized naming for examples (cv2_*, gcapture_*, blackfly_capture_*).
2022 - February added libcamera capture for Raspian Bullseye
2022 - January added queue as intialization option, updated cv2Capture
2021 - November moved queue into class
2021 - November added rtp server and client
2021 - November added mkvServer, wheel installation, cleanup
2021 - October added aviServer and multicamera example, PySpin trigger fix
2021 - September updated PySpin trigger out polarity setting  
2020 - Initial release  
```

## Example Camera Performance

### Sony IMX287 FLIR Blackfly S BFS-U3-04S2M

- 720x540 524fps
- auto_exposure off

### OV5647 OmniVision RasPi

- auto_exposure 0: auto, 1:manual
- exposure in microseconds
- Max Resolution 2592x1944
- YU12, (YUYV, RGB3, JPEG, H264, YVYU, VYUY, UYVY, NV12, BGR3, YV12, NV21, BGR4)
- 320x240 90fps
- 640x480 90fps
- 1280x720 60fps
- 1920x1080 6.4fps
- 2592x1944 6.4fps

### IMX219 Sony RasPi

- auto_exposure 0: auto, 1:manual
- exposure in microseconds
- Max Resolution 3280x2464
- YU12, (YUYV, RGB3, JPEG, H264, YVYU, VYUY, UYVY, NV12, BGR4)
- 320x240 90fps
- 640x480 90fps
- 1280x720 60fps
- 1920x1080 4.4fps
- 3280x2464 2.8fps

### ELP USB Camera RasPi

- MJPG
- 320x240 and 640/480, 120fps
- auto_exposure, can not figure out out in MJPG mode
- auto_exposure = 0 -> static exposure
- exposure is about (exposure value / 10) in ms
- WB_TEMP 6500

### Dell Internal USB

- 320x240, 30fps
- YUY2
- autoexposure ? 0.25, 0.74 -1. 0
- WB_TEMP 4600
- 1280x720, 30fps
- 620x480, 30fps
- 960x540, 30fps

## Pip upload

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
