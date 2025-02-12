###############################################################################
# Camera utility functions for opencv
#
# Probe the cameras, return indices, fourcc, default resolution etc.
#
# Urs Utzinger 
# 2021 Initial release
###############################################################################
import cv2 
import platform

def probeCameras(numcams: int = 10):
    '''
    Scans cameras and returns default fourcc, width and height
    '''
    # check for up to 10 cameras
    index = 0
    arr = []
    i = numcams
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            tmp = cap.get(cv2.CAP_PROP_FOURCC)
            fourcc = "".join([chr((int(tmp) >> 8 * i) & 0xFF) for i in range(4)])
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            arr.append({"index": index, "fourcc": fourcc, "width": width, "height": height})
        index += 1
        i -= 1
    return arr

def findCamera(numcams: int = 10, fourccSig = "\x16\x00\x00\x00", widthSig: int=640, heightSig: int=480):
    '''
    Identifies camera with given default fourcc, width and height
    '''
    camprops = probeCameras(numcams)
    score = 0
    camera_index = 0
    for i in range(len(camprops)):
        try: found_fourcc = 1 if camprops[i]['fourcc'] == fourccSig else 0            
        except: found_fourcc = 0
        try: found_width = 1  if camprops[i]['width']  == widthSig  else 0
        except: found_width =  0
        try: found_height = 1 if camprops[i]['height'] == heightSig else 0   
        except: found_height = 0
        tmp = found_fourcc+found_width+found_height
        if tmp > score:
            score = tmp
            camera_index = i
    return camera_index

def genCapture(configs, camera_index: int=0):
    # Create camera interface based on computer OS you are running
    # plat can be Windows, Linux, MaxOS
    plat = platform.system()
    if plat == 'Linux' and platform.machine() == "aarch64": # this is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    else:
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs, camera_index)
        
    return camera

