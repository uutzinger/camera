##########################################################################
# Testing of rtp stream display 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging
import time

window_name = 'RTP'

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("RTP")

# create camera interface
from camera.capture.rtpcapture import rtpCapture
logger.log(logging.INFO, "Starting Capture")

# On Linux, rtpCapture can use GI/GStreamer if installed.
# On Windows/macOS without GI/GStreamer, the pip-friendly fallback uses FFmpeg and
# requires an SDP file describing the RTP stream (payload type, codec, clock rate).
configs = {
    # For FFmpeg fallback:
    # - `rtp_sdp` must point to a file describing the stream (see examples/rtp_h264_pt96.sdp)
    # - `camera_res` (or `output_res`) is required so raw frame reads can be sized correctly
    'rtp_sdp': 'examples/rtp_h264_pt96.sdp',
    'camera_res': (640, 480),
}

# Port must match the SDP file (m=video <port> ...)
camera = rtpCapture(configs=configs, port=554)
logger.log(logging.INFO, "Getting Images")
camera.start()

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

while(cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0):
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, msg)

    if camera.buffer and camera.buffer.avail > 0:
        frame, ts_ms = camera.buffer.pull(copy=False)
        if frame is not None:
            cv2.imshow(window_name, frame)
    else:
        time.sleep(0.001)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.close(timeout=2.0)
cv2.destroyAllWindows()
