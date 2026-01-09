##########################################################################
# Blackfly capture + display example
#
# What this script does
# - Captures frames from a FLIR Blackfly camera via `blackflyCapture` (PySpin).
# - Displays frames at a throttled rate (`displayfps`) while capturing at full
#   camera rate (`fps`). This keeps UI responsive even at very high FPS.
# - Logs capture throughput and display throughput every ~5 seconds.
#
# Notes
# - Requires the FLIR Spinnaker SDK / PySpin to be installed.
# - Uses `blackflyCapture.measured_fps` for the measured capture rate.
##########################################################################
# Results
#    Capture FPS: 524.2
#    Display FPS: 32.2
#    CPU Usage: 6-7%
##########################################################################

from __future__ import annotations

import logging
import time
from queue import Empty

import cv2


def main() -> None:

    configs = {
        'camera_res': (720, 540),
        'exposure': 1750,          # microseconds, -1 = autoexposure
        'autoexposure': 0,         # 0,1
        'fps': 500,
        'binning': (1, 1),
        'offset': (0, 0),
        'adc': 8,                  # 8,10,12,14 bit
        'trigout': 2,              # -1 no trigger output
        'ttlinv': True,
        'trigin': -1,
        'output_res': (-1, -1),
        'flip': 0,
        'displayfps': 50,
    }

    display_interval = 0.0 if configs['displayfps'] >= configs['fps'] else (1.0 / configs['displayfps'])

    window_name = 'Camera'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 60)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('Blackfly')

    from camera.capture.blackflycapture import blackflyCapture

    logger.info('Starting Capture')
    camera = blackflyCapture(configs)
    camera.start()

    dps_measure_time = 5.0
    last_display = time.time()
    last_time = time.time()
    measured_dps = 0.0
    num_frames_received = 0
    num_frames_displayed = 0

    stop = False
    try:
        while not stop:
            current_time = time.time()

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, '%s', msg)

            try:
                (_frame_time, frame) = camera.capture.get(block=True, timeout=1.0)
                num_frames_received += 1
            except Empty:
                frame = None

            if (current_time - last_time) >= dps_measure_time:
                logger.debug('Status:Frames obtained per second:%s', num_frames_received / dps_measure_time)
                measured_dps = num_frames_displayed / dps_measure_time
                logger.debug('Status:Frames displayed per second:%s', measured_dps)
                num_frames_received = 0
                num_frames_displayed = 0
                last_time = current_time

            if frame is not None and (current_time - last_display) > display_interval:
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Capture FPS:{camera.measured_fps} [Hz]', textLocation0, font, fontScale, fontColor, lineType)
                cv2.putText(display_frame, f'Display FPS:{measured_dps} [Hz]', textLocation1, font, fontScale, fontColor, lineType)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord('q'):
                    stop = True
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                        stop = True
                except Exception:
                    stop = True

                last_display = current_time
                num_frames_displayed += 1

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
