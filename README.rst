Camera Util
===========

A collection of python threaded camera support routines for  

-  USB and laptop internal webcams 
-  RTSP streams 
-  MIPI CSI cameras (Raspberry Pi, Jetson Nano) 
-  FLIR blackfly USB camera

Save data as, HDF5, TIFF, AVI, MKV  

Works on most platforms: Windows, MacOS, Unix  

For detailed documentation please `Readme.md
<https://github.com/uutzinger/camera/blob/master/README.md>`_.

Example program using camera
----------------------------

.. code-block:: python

   import cv2
   import logging
   import time

   configs = {
    'camera_res'      : (1920, 1080),   # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
    'exposure'        : -6,             # depends on camera: e.g. -5 =(2^-5)=1/32, 
                                        # 0=auto, 1...max=frame interval in microseconds
    'autoexposure'    : 3.0,            # depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 30,             # fps
    'fourcc'          : "MJPG",         # cv2 camera only: MJPG, YUY2, YUYV
    'buffersize'      : -1,             # default is 4 for V4L2, max 10, 
    'output_res'      : (-1, -1),       # output resolution (-1=no change) 
    'flip'            : 0,              # 0=no rotation 
    'displayfps'      : 5               # frame rate for display server
   }

   if configs['displayfps'] >= configs['fps']: display_interval = 0
   else: display_interval = 1.0/configs['displayfps']

   # Open display window
   cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

   # Setting up logging
   logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
   logger = logging.getLogger("Main")

   # Setting up storage
   from camera.streamer.mkvstorageserver import mkvServer
   mkv = mkvServer("C:\\temp\\" + "Test.mkv", configs['fps'], configs['camera_res'])
   mkv.start()

   # Create camera interface 
   from camera.capture.cv2capture import cv2Capture
   camera = cv2Capture(configs, 0)
   camera.start()
   
   stop = False
   while(not stop):
      current_time = time.time()

      # Wait for new image
      (frame_time, frame) = camera.capture.get(block=True, timeout=None)
      # take care of camera log messages
      while not camera.log.empty():
         (level, msg)=camera.log.get_nowait()
         logger.log(level, msg)


      # Put image into storage queue
      if not mkv.queue.full():
         mkv.queue.put_nowait((frame_time, frame)) 
      # take care of storage log messages
      while not mkv.log.empty():
         (level, msg)=mkv.log.get_nowait()
         logger.log(level, msg)

      # Display
      if (current_time - last_display) >= display_interval:
         cv2.imshow('Camera', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'): stop = True
         if cv2.getWindowProperty('Camera', 0) < 0: stop = True
         last_display = current_time
  
      # Do other things 

   # Clean up
   camera.stop()
   mkv.stop()
   cv2.destroyAllWindows()

