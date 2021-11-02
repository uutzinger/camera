# Camera

## Overview
A collection of python threaded camera support routines for  
* USB and laptop internal webcams
* RTSP streams
* MIPI CSI cameras (Raspberry Pi, Jetson Nano)
* FLIR blackfly (USB)

Also support to save as  
* HD5  
* tiff  
* avi  

Supported OS  
* Windows
* MacOS
* Unix

The routines primarily use OpenCV or PySpin to interface with the camera.
The image acquisition runs in a background thread to achieve maximal frame rate and minimal latency.

This work is based on efforts from [Mark Omo](https://github.com/ferret-guy) and [Craig Post](https://github.com/cpostbitbuckets).

## Requirements

* PySpin for FLIR cameras
* opencv for USB, CSI cameras, RTSP network streams
  * Windows uses cv2.CAP_MSMF
  * Darwin uses cv2.CAP_AVFOUNDATION
  * Linux uses cv2.CAP_V4L2
  * Jetson Nano uses cv2.CAP_GSTREAMER
  * RTSP uses cv2.CAP_GSTREAMER

```cv2.CAP_GSTREAMER``` requires custom built opencv on Windows as gstreamer is not enabled by default. See my windows installation scripts on [Github](https://github.com/uutzinger/Windows_Install_Scripts). GSTREAMER is enabled by default on jetson nano opencv.

## Installation

1. ``` cd "folder where you have this Readme.md file" ```
2. ``` pip install . ``` or ```python setup.py install ```

### To install pre requisites on Windows:
**opencv** 

3. ```pip3 install opencv-contrib-python```  

**tiff** or **hd5**  
* ```https://www.lfd.uci.edu/~gohlke/pythonlibs/#imagecodecs```
* ```https://www.lfd.uci.edu/~gohlke/pythonlibs/#tifffile```
* ```https://www.lfd.uci.edu/~gohlke/pythonlibs/#h5py```   

matching your python installation (e.g. 3.8) and CPU architecture (e.g. 64).

**blackfly**  
Spinnaker provides SDK and python bindings. The versions of those two programs need to match.
* Spinnaker SDK is at https://flir.app.boxcn.net/v/SpinnakerSDK, install development options
* download spinnaker_python from same location 

Then in CMD window:  

4. ``` cd Downloads```   
5. ``` pip3 install imagecodecs....```  
6. ``` pip3 install tifffile....```   
7. ``` pip3 install h5py....```  
8. ``` pip3 install spinnaker_python...```

with ```....``` replaced by TAB-key.

9. Make sure you have ```C:\temp``` directory if you use the example storage programs.

10. To get best tiff performance intalling libtiff is advised: https://github.com/uutzinger/Windows_Install_Scripts/blob/master/Buildinglibtiff.md

### To install OpenCV on Raspi:  

3. ```cd ~```
4. ```sudo pip3 install opencv-contrib-python==4.1.0.25```
5. ```sudo pip3 install tifffile h5py platform imagecodecs```

## How to create config files
A. Specifications of your camera  

On Windows, the Camera utility will give you resolution options and frames per second.
To investigate other options you can use OBS studio (or any other capture program), establish camera capture device and inspect video options. 
When the `test_camer_display.py` is started and DEBUG logging is enabled, it will list all camera options the system offers. When an option states `-1` it likely is not available for that camera.

B. Configuration file  

Use one of the existing camera configutrations in ```examples/configs``` or create your own. 
As first step set appropriate resolution and frames per second. 
As second step figure out the exposure and autoexposure settings.

C. Run ```test_camera_display.py``` from ```.\examples```
You need to set the proper config file in this program. You should not need to edit python files in capture or streamer folder.

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
**Display**:   
In general display should occur in main program. 
OpenCV requires waitkey to be executed in order to update the display and limits update rate to about 50-90 fps.

**Queue**: 
Data transfer between the main program and capture and storage threads. 

**Examples**: 
* ```test_camera_display.py``` tests camera capture for all capture platforms except blackfly.  
* ```test_blackfly_display.py``` tests the blackfly capture module, displays images and reports framerate.  

* ```test_display.py``` testing of opencv display framerate, no camera, just refresh rate.
* ```test_savehd5.py``` testing of the disk throughput with hdf5, no camera
* ```test_savetiff.py``` testing of the disk throughput with tiff, no camera
* ```test_saveavi.py``` testing of the disk throughput with avi, no camera, only 3 color planes per image possible
* ```test_savemkv.py``` testing of the disk throughput with mkv/mp4v, no camera, only 3 color planes per image possible
* ```test_blackfly.py``` tests the blackfly capture module and reports framerate, no display

* ```test_saveavi_display.py``` example for multiple camera and save to avi files
* ```test_blackfly_savehdf5.py``` no display but incoporates saving to disk  
* ```test_blackfly_savetiff.py``` no display but incoporates saving to disk  
* ```test_blackfly_savehdf5_display.py``` display and incoporates saving to disk  
* ```test_blackfly_savetiff_display.py``` display andincoporates saving to disk  

## Changes
```
2021 - October added aviServer and multicamera example, PySpin trigger fix
2021 - September updated PySpin trigger out polarity setting  
2020 - Release  
Urs Utzinger
```

## References
realpython:  
https://realpython.com/python-concurrency/  
https://realpython.com/python-sleep/ 
https://realpython.com/async-io-python/  

python for the lab:  
https://www.pythonforthelab.com/blog/handling-and-sharing-data-between-threads/  


## Camera Settings
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
