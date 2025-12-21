##########################################################################
# Blackfly capture + process + display + store corrected cubes to HDF5
#
# Notes
# - Collects 14 frames into a cube.
# - Estimates a background frame as the minimum-intensity frame in the cube.
# - Applies a simple (frame - background) * flatfield correction into uint16.
# - Writes corrected cubes to HDF5 and displays a preview frame.
# - Output goes to %TEMP% on Windows or $TMPDIR (/tmp) on Linux.
##########################################################################

from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from numba import vectorize


def _default_output_dir() -> Path:
    if platform.system() == 'Windows':
        return Path(os.environ.get('TEMP', 'C:/temp'))
    return Path(os.environ.get('TMPDIR', '/tmp'))


@vectorize(['uint16(uint8, uint8, uint16)'], nopython=True, fastmath=True)
def calibrate(pixel: np.uint8, background: np.uint8, flatfield: np.uint16) -> np.uint16:
    value = int(pixel) - int(background)
    if value < 0:
        value = 0
    out = value * int(flatfield)
    if out > 65535:
        out = 65535
    return out


def main() -> None:

    from configs.blackfly_configs import configs

    display_interval = 0.0 if configs['displayfps'] >= configs['fps'] else (1.0 / configs['displayfps'])

    res = configs['camera_res']
    width, height = res[0], res[1]
    measure_time = 5.0

    depth = 14

    data_cube = np.zeros((depth, height, width), dtype=np.uint8)
    background = np.zeros((height, width), dtype=np.uint8)
    flatfield = np.asarray((2**8) * np.ones((height, width)), dtype=np.uint16)
    inten = np.zeros(depth, dtype=np.uint16)
    data_cube_corr = np.zeros((depth, height, width), dtype=np.uint16)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('BlackflyProcHDF5')

    from camera.streamer.h5storageserver import h5Server

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hdf5'
    hdf5 = h5Server(str(output_dir / filename))
    logger.info('Starting Storage Server: %s', output_dir / filename)
    hdf5.start()

    from camera.capture.blackflycapture import blackflyCapture

    logger.info('Starting Capture')
    camera = blackflyCapture(configs)
    camera.start()

    window_name = 'Camera'
    proc_window = 'Proc'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 480)
    textLocation1 = (10, 520)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(proc_window, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    num_cubes_stored = 0
    num_cubes_generated = 0
    last_time = time.perf_counter()
    last_display = time.perf_counter()
    num_frames_displayed = 0
    proc_time = 0.0

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, '%s', msg)
            while not hdf5.log.empty():
                (level, msg) = hdf5.log.get_nowait()
                logger.log(level, '%s', msg)

            try:
                (frame_time, frame) = camera.capture.get(block=True, timeout=1.0)
            except Empty:
                frame = None

            if frame is None:
                continue

            data_cube[frame_idx, :, :] = frame
            frame_idx += 1

            if frame_idx >= depth:
                frame_idx = 0
                num_cubes_generated += 1

                proc_start = time.perf_counter()
                np.sum(data_cube[:, ::64, ::64], axis=(1, 2), out=inten)
                frame_idx_bg = int(np.argmin(inten))
                background[:, :] = data_cube[frame_idx_bg, :, :]
                calibrate(data_cube, background, flatfield, out=data_cube_corr)
                proc_time += (time.perf_counter() - proc_start)

                try:
                    hdf5.queue.put_nowait((frame_time, data_cube_corr.copy()))
                    num_cubes_stored += 1
                except Exception:
                    pass

                img8 = (data_cube_corr[0, :, :] >> 8).astype(np.uint8)
                cv2.putText(img8, 'Proc', (10, 20), font, fontScale, fontColor, lineType)
                cv2.imshow(proc_window, img8)

            if (current_time - last_time) >= measure_time:
                if num_cubes_generated > 0:
                    logger.info('Status:process time:{:.2f}ms'.format(proc_time * 1000.0 / max(1, num_cubes_generated)))
                logger.info('Status:cubes generated per second:%s', num_cubes_generated / measure_time)
                logger.info('Status:cubes sent to storage per second:%s', num_cubes_stored / measure_time)
                logger.info('Status:frames displayed per second:%s', num_frames_displayed / measure_time)
                num_cubes_generated = 0
                num_cubes_stored = 0
                num_frames_displayed = 0
                proc_time = 0.0
                last_time = current_time

            if (current_time - last_display) >= display_interval:
                # display latest corrected preview (or raw frame if not yet computed)
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Capture FPS:{getattr(camera, "measured_fps", float("nan"))} [Hz]', textLocation0, font, fontScale, fontColor, lineType)
                cv2.putText(display_frame, f'Store CPS:{getattr(hdf5, "measured_cps", float("nan"))} [Hz]', textLocation1, font, fontScale, fontColor, lineType)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord('q'):
                    stop = True
                for name in (window_name, proc_window):
                    try:
                        if cv2.getWindowProperty(name, 0) < 0:
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
            hdf5.stop()
            hdf5.join(timeout=2.0)
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
