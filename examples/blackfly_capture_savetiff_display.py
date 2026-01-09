##########################################################################
# Blackfly capture + display + store cubes to TIFF
#
# Notes
# - Collects 14 frames into a cube and writes each cube to a multi-page TIFF.
# - Displays frames at a throttled rate.
# - Output goes to %TEMP% on Windows or $TMPDIR (/tmp) on Linux.
##########################################################################

from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full

import cv2
import numpy as np


def _default_output_dir() -> Path:
    if platform.system() == 'Windows':
        return Path(os.environ.get('TEMP', 'C:/temp'))
    return Path(os.environ.get('TMPDIR', '/tmp'))


def main() -> None:

    from configs.blackfly_configs import configs

    display_interval = 0.0 if configs['displayfps'] >= configs['fps'] else (1.0 / configs['displayfps'])

    dps_measure_time = 5.0

    width, height = configs['camera_res']
    data_cube = np.zeros((14, height, width), dtype=np.uint8)

    window_name = 'Camera'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, max(20, height - 60))
    textLocation1 = (10, max(40, height - 20))
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('BlackflyTIFF')

    from camera.streamer.tiffstorageserver import tiffServer

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.tiff'
    tiff = tiffServer(str(output_dir / filename))
    tiff.start()

    from camera.capture.blackflycapture import blackflyCapture

    logger.info('Starting Capture')
    camera = blackflyCapture(configs)
    camera.start()

    frame_idx = 0
    num_cubes_sent = 0
    num_cubes_generated = 0
    last_time = time.time()
    last_display = time.time()
    num_frames_displayed = 0
    measured_dps = 0.0

    frame = None

    stop = False
    try:
        while not stop:
            current_time = time.time()

            # Allow window-close even if capture stalls
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                    stop = True
            except Exception:
                stop = True

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, '%s', msg)
            while not tiff.log.empty():
                (level, msg) = tiff.log.get_nowait()
                logger.log(level, '%s', msg)

            try:
                (frame_time, frame) = camera.capture.get(block=True, timeout=1.0)
            except Empty:
                frame = None

            if frame is not None:
                data_cube[frame_idx, :, :] = frame
                frame_idx += 1

            if frame_idx >= 14:
                frame_idx = 0
                num_cubes_generated += 1

                try:
                    tiff.queue.put_nowait((frame_time, data_cube.copy()))
                    num_cubes_sent += 1
                except Full:
                    logger.warning('TIFF:Storage Queue is full!')
                except Exception as exc:
                    logger.warning('TIFF:Failed to enqueue cube: %s', exc)

            if (current_time - last_time) >= dps_measure_time:
                measured_cps_generated = num_cubes_generated / dps_measure_time
                measured_cps_sent = num_cubes_sent / dps_measure_time
                measured_dps = num_frames_displayed / dps_measure_time
                logger.debug('Status:cubes generated per second:%s', measured_cps_generated)
                logger.debug('Status:cubes sent to storage per second:%s', measured_cps_sent)
                logger.debug('Status:Frames displayed per second:%s', measured_dps)
                num_cubes_generated = 0
                num_cubes_sent = 0
                num_frames_displayed = 0
                last_time = current_time

            if frame is not None and (current_time - last_display) >= display_interval:
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Capture FPS:{getattr(camera, "measured_fps", float("nan"))} [Hz]', textLocation0, font, fontScale, fontColor, lineType)
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
        try:
            tiff.stop()
            tiff.join(timeout=2.0)
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
