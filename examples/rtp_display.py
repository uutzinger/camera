##########################################################################
# Testing of rtp stream display 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging

window_name = 'RTP'

# Setting up logging
logging.basicConfig(level=logging.DEBUG)

# reate camera interface
from camera.capture.rtpcapture import rtpCapture
print("Starting Capture")

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
print("Getting Images")
camera.start()

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

while(cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0):
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.stop()
cv2.destroyAllWindows()
