Camera Util
===========

A collection of threaded camera capture and streaming utilities.

Supports:

- USB and laptop internal webcams (OpenCV)
- RTSP/RTP network streams (GStreamer or FFmpeg fallback)
- MIPI CSI cameras (Raspberry Pi Picamera2/libcamera, Jetson Nano)
- Teledyne/FLIR Blackfly (PySpin)

Optional storage helpers for HDF5, TIFF, AVI, MKV.

Full documentation (configs, backends, examples): `README.md <https://github.com/uutzinger/camera/blob/master/README.md>`_.

Install
-------

.. code-block:: bash

   pip install camera-util

Quickstart (OpenCV webcam)
--------------------------

.. code-block:: python

   import logging
   import time
   import cv2
   from camera.capture.cv2capture import cv2Capture

   configs = {
       "camera_res": (640, 480),
       "fps": 30,
       "fourcc": "MJPG",
       "displayfps": 30,
   }

   camera = cv2Capture(configs, 0)
   camera.start()

   window_name = "Camera"
   cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
   last_display = time.perf_counter()
   display_interval = 1.0 / configs["displayfps"]
   stop = False
   while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0 and not stop:
       current_time = time.perf_counter()
       frame_time, frame = camera.capture.get(block=True, timeout=None)
       if (current_time - last_display) >= display_interval:
           frame_display = frame.copy()
           cv2.putText(frame_display, f"Capture FPS:{camera.measured_fps:.1f} [Hz]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
           cv2.imshow(window_name, frame_display)
           if cv2.waitKey(1) & 0xFF == ord("q"):
               stop = True
           last_display = current_time

   camera.stop()
   cv2.destroyAllWindows()

Raspberry Pi (Picamera2/libcamera)
----------------------------------

Two wrappers are provided:

- ``piCamera2Capture`` (threaded, non-Qt)
- ``piCamera2CaptureQt`` (Qt5/Qt6)

Frame delivery model:

- Frames go into a single-producer/single-consumer ring buffer ``camera.buffer``.
- Consumers poll (no queue semantics):

.. code-block:: python

   if camera.buffer and camera.buffer.avail > 0:
       frame, ts_ms = camera.buffer.pull(copy=False)

The Qt wrapper does not emit a per-frame signal; GUI code typically polls via a ``QTimer``.

