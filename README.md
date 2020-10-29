# Camera

## Overview
A collection of python threaded camera support routines for  
* USB and internal webcams
* rtsp streams
* MIPI CSI cameras (Raspberry Pi, Jetson Nano)
* FLIR blackfly (USB 3)

Supported OS  
* Windows
* MacOS
* Linux

The routines primarily use OpenCV or PySpin to interface with the camera.
The main effortt here is to run the image acquisition in a background thread and to find best approaches for maximal frame rate and minimal latency.  
This work includes efforts from Mark Omo and Craig Post.  

Urs Utzinger, 2020

## Requirements
```
PySpin for FLIR cameras
   Any OS:      blackflyCapture  
cv2 for USB and CSI cameras  
   Windows:     cv2Capture:  cv2.CAP_MSMF  
   Darwin:      cv2Capture:  cv2.CAP_AVFOUNDATION  
   Linux:       cv2Capture:  cv2.CAP_V4L2  
   Jetson Nano: nanoCapture: cv2.CAP_GSTREAMER  
   Other:       cv2Capture:  cv2.CAP_ANY
RTSP network streams
   Any[*]:      cv2Capture:  cv2.CAP_GSTREAMER
cv2 for image resizing and flipping    
```
[*] RTSP requires gstreamer integration. CV2 will need to be custom built on windows to enable gstreamer support. See my windows installation scripts on Github.

## Capture modules

### **blackflyCapture**
Supports all settings needed for Blackfly camera.   
Supports trigger out during frame exposure and trigger in for frame start.  
Optimized settings to achieve full frame rate (520Hz) with S BFS-U3-04S2M.

### **nanoCapture**
Uses gstreamer pipeline for Jetson Nano.
Optimized pipline for nvidia conversion and nvarguscamera capture.  
Settings optimized for Sony IMX219 Raspi v2 Module.

### **cv2Capture**
Uses the cv2 capture architecture. THe video subsystem is choosen based on the operating system.

### **rtspCapture**
gstreamer based rtsp network stream capture for all platforms.  
gstreamer is called through OpenCV. By default OpenCV supports ffmpeg and not gstreamer. Jetson Nano does not support ffmpeg.

### **piCapture**
Interface for picamera module. Depricated since cv2Capture is more efficient for the Raspberry Pi.

## Example Programs
test_blackfly.py tests the blackfly capture module, displays images and reports framerate.  
test_blackfly_savehdf5.py same as above but incoporates saving to disk.  
test_camera.py unifying camera capture for all capture platforms except blackfly.  
test_rtsp.py testing rtsp network streams.  

## Camera Settings
Configs folder:  
```
  FLIR Blackfly S BFS-U3-04S2M: .. blackfly_configs.py  
  Raspi v1 OV5647: ............... raspy_v1module_configs.py  
  Raspi v2 IMX219: ............... raspy_v2module_configs.py  
  Dell internal: ................. dell_internal_confgis.py  
  ELP USB: ....................... configs.py  
````
### Sony IMX287 FLIR Blackfly S BFS-U3-04S2M
* 720x540 524fps
* auto_exposure off

### OV5647 OmniVision RasPi
* auto_exposure 0: auto, 1:manual
* exposure in microseconds
* Max Resolution 2592x1944
* YU12, (YUYV, RGB3, JPEG, H264, YVYU, VYUY, UYVY, NV12, BGR3, YV12, NV21, BGR4)
* 320x240 90fps
* 640x480 90fps
* 1280x720 60fps
* 1920x1080 6.4fps
* 2592x1944 6.4fps

### IMX219 Sony RasPi
* auto_exposure 0: auto, 1:manual
* exposure in microseconds
* Max Resolution 3280x2464
* YU12, (YUYV, RGB3, JPEG, H264, YVYU, VYUY, UYVY, NV12, BGR4)
* 320x240 90fps
* 640x480 90fps
* 1280x720 60fps
* 1920x1080 4.4fps
* 3280x2464 2.8fps

### ELP USB Camera
* MJPG
* 320x240, 120fps
* auto_exposure ?
* WB_TEMP 6500

### Dell Internal USB
* 320x240, 30fps
* YUY2
* autoexposure ? 0.25, 0.74 -1. 0
* WB_TEMP 4600
* 1280x720, 30fps
* 620x480, 30fps
* 960x540, 30fps
