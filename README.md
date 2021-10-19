# Camera

## Overview
A collection of python threaded camera support routines for  
* USB and internal webcams
* RTSP streams
* MIPI CSI cameras (Raspberry Pi, Jetson Nano)
* FLIR blackfly (USB)

Supported OS  
* Windows
* MacOS
* Linux

The routines primarily use OpenCV or PySpin to interface with the camera.
The main effortt with these routines is to run the image acquisition in a background thread and to find best approaches for maximal frame rate and minimal latency.  
This work is based efforts from [Mark Omo](https://github.com/ferret-guy) and [Craig Post](https://github.com/cpostbitbuckets).

2020 Release  
2021 Updated PySpin trigger out polarity setting  
2021 Added avi server and multicamera example  
Urs Utzinger

## References
realpython:  
https://realpython.com/python-concurrency/  
https://realpython.com/python-sleep/ 
https://realpython.com/async-io-python/  
python for the lab  
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
[*] RTSP requires gstreamer integration. CV2 will need to be custom built on windows to enable gstreamer support. See my windows installation scripts on [Github](https://github.com/uutzinger/Windows_Install_Scripts).

To install opencv on Windows:
* ```pip3 install opencv-python```
* ```pip3 install opencv-contrib-python```  
* ```pip3 install tifffile h5py platform```  

Make sure you have ```C:\temp directory``` if you use storage server.

To install opencv on Raspi:  
(probalby dont need all packages...)
```
cd ~
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get -y install python3-pybind11
sudo apt-get -y install libusb-1.0-0-dev
sudo apt-get -y install swig
sudo apt-get -y install gfortran
sudo apt-get -y install python3-numpy python3-dev python3-pip python3-mock
sudo apt-get -y install libjpeg-dev libtiff-dev libtiff5-dev libjasper-dev libpng-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libavresample-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install libhdf5-dev libhdf5-serial-dev
sudo apt-get -y install libopenblas-dev liblapack-dev libatlas-base-dev libblas-dev  libeigen{2,3}-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo pip3 install --upgrade setuptools
sudo pip3 install opencv-contrib-python==4.1.0.25
pip3 install tifffile h5py platform
```

## Capture modules

### **blackflyCapture**
Simplifies the settings needed for the Blackfly camera.   
Supports trigger out during frame exposure and trigger in for frame start.  
Optimized settings to achieve full frame rate with S BFS-U3-04S2M.

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
Interface for picamera module. Depricated since cv2Capture is more efficient for the Raspberry Pi.

## Example Programs
**Display**: In general display should occur in main program. OpenCV requires waitkey to be executed in order to update the display. Waikey takes much longer than 1ms and therfore transferring data between threads is significantly slowed down in the main thread if display is requested.

**Queue**: Data transfer between threads or between main program and thread works better with Queue than with setting new data falgs and accessing it through shared memory. Queue can be programmed to be blocking or non blocking and if the queue size is long enough, no data is lost if the main thread can  not keep up with the capture thread for brief amount of time.

* ```test_blackfly.py``` tests the blackfly capture module and reports framerate.
* ```test_blackfly_display.py``` tests the blackfly capture module, displays images and reports framerate.  
* ```test_blackfly_savehdf5.py``` same as above, no display but incoporates saving to disk.  
* ```test_camera.py``` unifying camera capture for all capture platforms except blackfly.  
* ```test_rtsp.py``` testing rtsp network streams.

* ```test_display.py``` testing opencv display framerate, no camera, just static refresh rate
* ```test_savehd5.py``` testing the disk throughput with hdf5, no camera
* ```test_sum.py``` testing different approaches to calculate the integreal/brightness of an image  
* ```test_arraycopy.py``` testing which axis in 3D arrays should be used for time  

* ```test_saveravi_display.py``` example for multiple camera and save to avi file

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
