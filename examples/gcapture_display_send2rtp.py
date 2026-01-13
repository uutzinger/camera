##########################################################################
# Testing of gCapture to RTP (display + send)
##########################################################################

import cv2
import logging
import time

from queue import Empty


# default camera starts at 0 by operating system
camera_index = 0

window_name = 'gCapture RTP'
font = cv2.FONT_HERSHEY_SIMPLEX
textLocation = (10, 20)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


# Select config
from configs.eluk_configs import configs as configs
# from configs.nano_IMX219_configs  import configs as configs
# from configs.raspi_v1module_configs  import configs as configs
# from configs.raspi_v2module_configs  import configs as configs


if configs['displayfps'] >= configs['fps']:
    rtp_interval = 1.0 / configs['fps']
    rtp_fps = configs['fps']
else:
    rtp_interval = 1.0 / configs['displayfps']
    rtp_fps = configs['displayfps']

rtp_size = configs.get('output_res', (-1, -1))
if rtp_size[0] <= 0 or rtp_size[1] <= 0:
    rtp_size = configs['camera_res']


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("gCapture2rtp")


# RTP streamer
from camera.streamer.rtpserver import rtpServer

logger.log(logging.INFO, "Starting RTP Server")
rtp = rtpServer(resolution=rtp_size, fps=rtp_fps, host='127.0.0.1', port=554, bitrate=2048, gpu=False, sdp_path='rtp_554.sdp')
while not rtp.log.empty():
    (level, msg) = rtp.log.get_nowait()
    logger.log(level, msg)
rtp.start()


# GStreamer capture
from camera.capture.gcapture import gCapture

camera = gCapture(configs, camera_num=camera_index)
logger.log(logging.INFO, "Getting Images")
while not camera.log.empty():
    (level, msg) = camera.log.get_nowait()
    logger.log(level, msg)
camera.start()


last_rtp = time.time()
stop = False
frame = None

while not stop:
    try:
        (frame_time, frame) = camera.capture.get(timeout=0.25)
    except Empty:
        frame_time = None

    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, msg)

    # display and transmit
    current_time = time.time()
    if (frame is not None) and ((current_time - last_rtp) >= rtp_interval):
        last_rtp = current_time

        frame_rtp = frame.copy()
        cv2.putText(frame_rtp, "Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation, font, fontScale, fontColor, lineType)

        if not rtp.queue.full():
            rtp.queue.put_nowait((frame_time, frame_rtp))
        else:
            logger.log(logging.WARNING, "Status:rtp Queue is full!")

        cv2.imshow(window_name, frame_rtp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
            stop = True

    while not rtp.log.empty():
        (level, msg) = rtp.log.get_nowait()
        logger.log(level, msg)


camera.stop()
try:
    camera.join(timeout=2.0)
except Exception:
    pass
try:
    camera.close_cam()
except Exception:
    pass
rtp.stop()
cv2.destroyAllWindows()
