"""Configuration templates for all capture backends.

This file is meant as a reference / starting point.

Notes
- Keys listed here are based on what each capture backend reads from the passed-in
  `configs` dict.
- Some backends treat missing keys as an error (KeyError). Others provide
  defaults via `.get(...)`.
- Units:
  - `exposure` is treated as microseconds across this repo.
  - For GStreamer nvargus, that value is translated to nanoseconds internally.

"""

# Shared options (used by most capture modules)
# ---------------------------------------------------------------------------
common_configs = {
    ##############################################
    # Camera Settings (common)
    ##############################################
    'camera_res': (640, 480),          # capture width & height
    'exposure': -1,                    # microseconds; <=0 means auto-exposure (backend dependent)
    'fps': 30,                         # requested capture FPS (backend dependent)

    ##############################################
    # Output / Display (common)
    ##############################################
    'output_res': (-1, -1),            # output width & height; (-1,-1) means do not resize
    'flip': 0,                         # 0..7, shared enum across modules:
                                       # 0=norotation
                                       # 1=ccw90deg
                                       # 2=rotation180
                                       # 3=cw90
                                       # 4=horizontal
                                       # 5=upright diagonal flip (ccw90 + horizontal)
                                       # 6=vertical
                                       # 7=upperleft diagonal flip (transpose)

    # Used by many example scripts / test loops (not always required by capture classes)
    'displayfps': 30,                  # display loop FPS; examples often skip frames for display
}


# OpenCV / V4L2 camera capture (Thread)
# ---------------------------------------------------------------------------
# Backend: camera/capture/cv2capture.py (cv2Capture)
cv2capture_configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res': (640, 480),
    'exposure': -1,                    # microseconds-ish; OpenCV backends vary
    'autoexposure': -1,                # -1: do not touch; otherwise best-effort
                                       # semantic intent: 0/False=manual, 1/True=auto
    'fps': -1,                         # -1: do not request fps
    'fourcc': -1,                      # -1: leave as-is; else int FOURCC or 4-char string (e.g. 'MJPG')
    'buffersize': 1,                   # CAP_PROP_BUFFERSIZE (backend dependent)
    'gain': -1,                        # CAP_PROP_GAIN
    'wb_temp': -1,                     # CAP_PROP_WB_TEMPERATURE
    'autowb': -1,                      # CAP_PROP_AUTO_WB
    'settings': -1,                    # if > -1: request opening backend camera settings dialog (if supported)

    ##############################################
    # Output / Display
    ##############################################
    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# OpenCV camera capture (multiprocessing)
# ---------------------------------------------------------------------------
# Backend: camera/capture/cv2captureProc.py (cv2CaptureProc)
cv2capture_proc_configs = {
    'camera_res': (640, 480),
    'exposure': -1,
    'autoexposure': -1,
    'fps': -1,
    'fourcc': -1,                      # int or 4-char string
    'buffersize': 1,

    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# Generic GStreamer capture via PyGObject appsink (cross-platform)
# ---------------------------------------------------------------------------
# Backend: camera/capture/gcapture.py (gCapture)
# This is the primary "direct GStreamer" backend (no OpenCV VideoCapture).
#
# Source selection priority:
#   1) gst_source_str / gst_source_pipeline (full custom source string)
#   2) gst_source or source (one of: 'auto', 'libcamera', 'nvargus', 'v4l2')
#   3) internal auto-detect (libcamerasrc -> nvarguscamerasrc -> v4l2src)

gcapture_configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res': (1280, 720),
    'exposure': -1,                    # microseconds; <=0 means auto
    'fps': 30,

    # Source selection / customization
    'gst_source': 'auto',              # or use 'source'
    'source': None,                    # alias for gst_source
    'gst_source_str': None,            # full source snippet (must end before downstream videoconvert)
    'gst_source_pipeline': None,       # alias for gst_source_str
    'gst_source_props_str': None,      # appended after element name, e.g. 'do-timestamp=true'

    # v4l2-only convenience
    'device': None,                    # e.g. '/dev/video0'
    'v4l2_device': None,               # alias for device

    ##############################################
    # Output / Display
    ##############################################
    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# Jetson Nano CSI capture via PyGObject appsink (nvarguscamerasrc default)
# ---------------------------------------------------------------------------
# Backend: camera/capture/nanocapture.py (nanoCapture)
# Same overall shape as gCapture, but defaults to nvargus on Jetson.

nanocapture_configs = {
    'camera_res': (1280, 720),
    'exposure': -1,                    # microseconds; <=0 means auto
    'fps': 30,

    # Source selection / customization
    'gst_source': 'nvargus',           # or 'v4l2' or 'libcamera' etc.
    'source': 'nvargus',               # alias
    'gst_source_str': None,
    'gst_source_pipeline': None,
    'gst_source_props_str': None,

    # v4l2-only convenience
    'device': None,
    'v4l2_device': None,

    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# Raspberry Pi CSI capture via Picamera2/libcamera
# ---------------------------------------------------------------------------
# Backend: camera/capture/picamera2capture.py (piCamera2Capture)

picamera2_configs = {
    'camera_res': (640, 480),
    'exposure': -1,                    # microseconds; <=0 means auto
    'autoexposure': -1,                # best-effort; mapped to libcamera controls where possible
    'fps': 30,

    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# Raspberry Pi CSI capture via legacy picamera (PiCamera)
# ---------------------------------------------------------------------------
# Backend: camera/capture/picapture.py (piCapture)

picamera_legacy_configs = {
    'camera_res': (640, 480),
    'exposure': -1,                    # microseconds; <=0 means auto (picamera shutter_speed=0)
    'fps': 30,

    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# RTSP stream capture via OpenCV + GStreamer
# ---------------------------------------------------------------------------
# Backend: camera/capture/rtspcapture.py (rtspCapture)

rtspcapture_configs = {
    'rtsp': 'rtsp://user:pass@host:554/stream',

    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# RTP (UDP) point-to-point stream capture via OpenCV + GStreamer
# ---------------------------------------------------------------------------
# Backend: camera/capture/rtpcapture.py (rtpCapture)
# Note: `port` is passed as an argument to the constructor, not via configs.

rtpcapture_configs = {
    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 30,
}


# FLIR Blackfly capture via PySpin
# ---------------------------------------------------------------------------
# Backend: camera/capture/blackflycapture.py (blackflyCapture)

blackfly_configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res': (720, 540),          # width & height (ROI)
    'exposure': 1750,                  # microseconds; <=0 means auto depending on autoexposure
    'autoexposure': 0,                 # 0/1, camera-specific
    'fps': 500,
    'binning': (1, 1),                 # (horizontal, vertical)
    'offset': (0, 0),                  # (x, y) ROI offset
    'adc': 8,                          # 8/10/12/14 bit

    # Triggering / IO
    'trigout': -1,                     # -1 no trigger output, else line number
    'ttlinv': False,                   # invert TTL logic
    'trigin': -1,                      # -1 software trigger, else line number

    ##############################################
    # Output / Display
    ##############################################
    'output_res': (-1, -1),
    'flip': 0,
    'displayfps': 50,
}
