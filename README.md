# Camera

## Overview
A collection of python threaded camera support routines for  
* USB and laptop internal webcams
* RTSP streams
* MIPI CSI cameras (Raspberry Pi, Jetson Nano)
* FLIR blackfly (USB)

Supported OS  
* Windows
* MacOS
* Linux, Raspian

The routines primarily use OpenCV or PySpin to interface with the camera.
The image acquisition runs in a background thread to achieve maximal frame rate and minimal latency.

This work is based on efforts from [Mark Omo](https://github.com/ferret-guy) and [Craig Post](https://github.com/cpostbitbuckets).

```
2021 - October added aviServer and multicamera example, PySpin trigger fix
2021 - September updated PySpin trigger out polarity setting  
2020 - Release  
```
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
[*] RTSP requires gstreamer integration. CV2 will need to be custom built on windows to enable gstreamer support. See my windows installation scripts on [Github](https://github.com/uutzinger/Windows_Install_Scripts) if RTSP functionaliy is needed.

## Installation

```
cd "folder where you have this Readme.md file"
pip install .
or
python setup.py install
or
py -3 setup.py install

```
### To install standard opencv on Windows:
* ```pip3 install opencv-python```
* ```pip3 install opencv-contrib-python```  

donwload from https://www.lfd.uci.edu/~gohlke/pythonlibs/   
* ```https://www.lfd.uci.edu/~gohlke/pythonlibs/#imagecodecs```
* ```https://www.lfd.uci.edu/~gohlke/pythonlibs/#tifffile```
* ```https://www.lfd.uci.edu/~gohlke/pythonlibs/#h5py```

then in CMD window with .... repalced through autocompleting TAB.
```
cd Downloads
pip3 install imagecodecs....
pip3 install tifffile....
pip3 install h5py....
```

Make sure you have ```C:\temp``` direcory if you use the example storage programs.

### To install OpenCV on Raspi:  
(probalby dont need all the packages in the middle)
```
cd ~
sudo apt-get -y update
sudo apt-get -y upgrade

sudo apt-get -y install cmake gfortran
sudo apt-get -y install python3-pybind11
sudo apt-get -y install libusb-1.0-0-dev
sudo apt-get -y install swig
sudo apt-get -y install python3-numpy python3-dev python3-pip python3-mock
sudo apt-get -y install libjpeg-dev libtiff-dev libtiff5-dev libjasper-dev libpng-dev libgif-dev libhdf5-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libavresample-dev  libdc1394-22-de
sudo apt-get -y install libxvidcore-dev libx264-dev 
sudo apt-get -y install libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev libblas-dev  libeigen{2,3}-dev
sudo apt-get -y insyall libgtk-3-dev 
sudo apt-get -y insyall libtbb2 libtbb-dev
sudo apt-get install libjasper-dev 
sudo apt-get install protobuf-compiler

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
sudo pip3 install --upgrade setuptools
sudo pip3 install opencv-contrib-python==4.1.0.25
sudo pip3 install tifffile h5py platform imagecodecs
```
## How to use
1. Take a look at the specifications of your camera. 

If you use USB camera on windows you can use Window Camera utility to figure out resolution options and frames per second. To investigate other options you can use OSB studio, establish camera capture device and look into video options.

2. You will need to create a configuration file.

Use one of the existing camera configutrations in ```examples/configs``` or create your own one. As first step set appropriate resolution and frames per second. As second step figure out the exposure and autoexposure settings.

3. Then start with a program in ```.\examples``` such as ```test_camera.py```. 

You should not need to edit python files in  capture or streamer folder.

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

**Queue**: Data transfer between threads or between main program and thread works better with Queue than with setting new data falgs and accessing it through shared memory. Queue can be programmed to be blocking or non blocking and if the queue size is long enough, no data is lost if a thread can not keep up with the capture thread for brief amount of time.

* ```test_blackfly.py``` tests the blackfly capture module and reports framerate.
* ```test_blackfly_display.py``` tests the blackfly capture module, displays images and reports framerate.  
* ```test_blackfly_savehdf5.py``` same as above, no display but incoporates saving to disk.  
* ```test_camera.py``` unifying camera capture for all capture platforms except blackfly.  
* ```test_rtsp.py``` testing rtsp network streams.

* ```test_display.py``` testing opencv display framerate, no camera, just static refresh rate
* ```test_savehd5.py``` testing the disk throughput with hdf5, no camera
* ```test_sum.py``` testing different approaches to calculate the integreal/brightness of an image  
* ```test_arraycopy.py``` testing which axis in 3D arrays should be used for time  

* ```test_saveavi_display.py``` example for multiple camera and save to avi files

## Camera Settings
Configs folder:  
```
  FLIR Blackfly S BFS-U3-04S2M: .. blackfly_configs.py  
  Raspi v1 OV5647: ............... raspy_v1module_configs.py  
  Raspi v2 IMX219: ............... raspy_v2module_configs.py  
  Dell internal: ................. dell_internal_confgis.py
  Eluktronics internal: .......... eluk_configs.py  
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
