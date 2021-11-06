Camera Util
===========

A collection of python threaded camera support routines for  

-  USB and laptop internal webcams 
-  RTSP streams 
-  MIPI CSI cameras (Raspberry Pi, Jetson Nano) 
-  FLIR blackfly (USB)

Save data  as  

-  HD5  
-  tiff  
-  avi  

Works on  

-  Windows  
-  MacOS  
-  Unix  

Documentation
-------------
For detailed documentation please `Readme.md
<https://github.com/uutzinger/camera/blob/master/README.md>`_.

Example Program using camera
----------------------------

.. code-block:: python

   import cv2
   import logging
   import time
   from queue import Queue

   configs = {
    'camera_res'      : (1920, 1080),   # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
    'exposure'        : -6,             # depends on camera: e.g. -5 =(2^-5)=1/32, 0 = auto, 1...max=frame interval in microseconds
    'autoexposure'    : 3.0,            # depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 30,             # fps
    'fourcc'          : "MJPG",         # cv2 camera only: MJPG, YUY2, YUYV
    'buffersize'      : -1,             # default is 4 for V4L2, max 10, 
    'output_res'      : (-1, -1),       # output resolution (-1 = no change) 
    'flip'            : 0,              # 0 = norotation 
    'displayfps'      : 5               # frame rate for display server
   }

   if configs['displayfps'] >= configs['fps']: display_interval = 0
   else: display_interval = 1.0/configs['displayfps']

   cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

   # Setting up logging
   logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING

   # Setting up input and/or output Queue
   captureQueue = Queue(maxsize=32)
   storageQueue = Queue(maxsize=64)

   # Setting up storage
   from camera.streamer.mkvstorageserver import mkvServer
   mkv = mkvServer("C:\\temp\\" + "Test.mkv", configs['fps'], configs['camera_res'])
   mkv.start(storageQueue)

   # Create camera interface 
   from camera.capture.cv2capture import cv2Capture
   camera = cv2Capture(configs, 0)
   camera.start(captureQueue)

   while(cv2.getWindowProperty('Camera', 0) >= 0):
      current_time = time.time()

      # Wait for new image
      (frame_time, frame) = captureQueue.get(block=True, timeout=None)
      # put image into storage queue
      if not storageQueue.full():
         storageQueue.put((frame_time, frame), block=False) 

      # Display
      if (current_time - last_display) >= display_interval:
         cv2.imshow('Camera', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
               break
         last_display = current_time
      
      # Do other things 

   # Clean up
   camera.stop()
   cv2.destroyAllWindows()

