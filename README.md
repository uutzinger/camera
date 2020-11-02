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

## References
https://realpython.com/python-concurrency/  
https://realpython.com/python-sleep/ 
https://realpython.com/async-io-python/  
https://www.pythonforthelab.com/blog/handling-and-sharing-data-between-threads/  

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
In general display should occur in main program. OpenCV requires waitkey to be executed in order to update the display. Waikey takes much longer than 1ms and therfore transferring data between threads is significantly slowed down in the main thread if display is requested.

Data transfer between threads or between main program and thread works better with Queue than with checking wether new data is available and then accessing it through shared memory. That is because Queue can be programmed to be blocking or non blocking and if Queue is long enough, not data is lost if main thread could not keep up with capture thread for brief amount of time.

test_blackfly.py tests the blackfly capture module and reports framerate.  
test_blackfly_display.py tests the blackfly capture module, displays images and reports framerate.  
test_blackfly_savehdf5.py same as above, no display but incoporates saving to disk.  
test_camera.py unifying camera capture for all capture platforms except blackfly.  
test_rtsp.py testing rtsp network streams.

test_display.py testing opencv display framerate  
test_savehd5.py testing the disk throughput with hdf5  
test_sum.py testing different approaches to calculate the integreal of an image  
test_arraycopy.py testing whihch axis should be used for the time  

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
