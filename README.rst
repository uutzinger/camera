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

   import cv2
   import time

   from camera.capture.cv2capture import cv2Capture

   configs = {
       "camera_res": (640, 480),
       "fps": 30,
       "fourcc": "MJPG",
   }

   camera = cv2Capture(configs, 0)
   camera.start()

   try:
       cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
       while cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) >= 0:
           frame_time, frame = camera.capture.get(timeout=0.25)
           if frame is not None:
               cv2.imshow("Camera", frame)
           if cv2.waitKey(1) & 0xFF == ord("q"):
               break
           time.sleep(0.001)
   finally:
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

   if camera.buffer and camera.buffer.avail() > 0:
       frame, ts_ms = camera.buffer.pull(copy=False)

The Qt wrapper does not emit a per-frame signal; GUI code typically polls via a ``QTimer``.

