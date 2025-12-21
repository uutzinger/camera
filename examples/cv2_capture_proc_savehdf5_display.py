##########################################################################
# Capture + process + display + store corrected cubes to HDF5 (cv2Capture)
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
def _correct_pixel(pixel: np.uint8, background: np.uint8, flatfield: np.uint16) -> np.uint16:
    value = int(pixel) - int(background)
    if value < 0:
        value = 0
    out = value * int(flatfield)
    if out > 65535:
        out = 65535
    return out


def main() -> None:

    configs = {
        'camera_res': (1280, 720),
        'exposure': -6,
        'autoexposure': 1.0,
        'fps': 30,
        'fourcc': -1,
        'buffersize': -1,
        'output_res': (-1, -1),
        'flip': 0,
        'displayfps': 30,
    }

    display_interval = 0.0 if configs['displayfps'] >= configs['fps'] else (1.0 / configs['displayfps'])

    width, height = configs['camera_res']
    measure_time = 5.0
    camera_index = 0

    depth = 14

    data_cube = np.zeros((depth, height, width), dtype=np.uint8)
    background = np.zeros((height, width), dtype=np.uint8)
    flatfield = np.asarray((2**8) * np.ones((height, width)), dtype=np.uint16)
    inten = np.zeros(depth, dtype=np.uint16)
    data_cube_corr = np.zeros((depth, height, width), dtype=np.uint16)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('cv2_capture_proc_savehdf5_display')

    from camera.streamer.h5storageserver import h5Server

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hdf5'
    hdf5 = h5Server(str(output_dir / filename))
    logger.info('Starting HDF5 storage: %s', output_dir / filename)
    hdf5.start()

    logger.info('Starting Capture')
    plat = platform.system()
    if plat == 'Linux' and platform.machine() == 'aarch64':
        from camera.capture.nanocapture import nanoCapture

        camera = nanoCapture(configs, camera_index)
    else:
        from camera.capture.cv2capture import cv2Capture

        camera = cv2Capture(configs, camera_index)

    camera.start()

    window_name = 'Main'
    proc_window = 'Proc'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, height - 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(proc_window, cv2.WINDOW_AUTOSIZE)

    frame_idx = 0
    num_cubes_stored = 0
    num_cubes_generated = 0

    last_stats = time.perf_counter()
    last_display = time.perf_counter()
    num_frames_displayed = 0

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            # Capture
            try:
                (frame_time, frame) = camera.capture.get(block=True, timeout=1.0)
            except Empty:
                frame = None

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, 'Status:%s', msg)

            if frame is None:
                # Still allow window-close / stop
                for name in (window_name, proc_window):
                    try:
                        if cv2.getWindowProperty(name, 0) < 0:
                            stop = True
                    except Exception:
                        stop = True
                continue

            data_cube[frame_idx, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1

            if frame_idx >= depth:
                frame_idx = 0
                num_cubes_generated += 1

                # Background estimation
                np.sum(data_cube[:, ::64, ::64], axis=(1, 2), out=inten)
                frame_idx_bg = int(np.argmin(inten))
                background[:, :] = data_cube[frame_idx_bg, :, :]

                # Correct cube
                _correct_pixel(data_cube, background, flatfield, out=data_cube_corr)

                # Show one corrected frame (downscale back to 8-bit for display)
                img8 = (data_cube_corr[0, :, :] >> 8).astype(np.uint8)
                cv2.putText(img8, f'Cube:{num_cubes_generated}', (10, 20), font, fontScale, fontColor, lineType)
                cv2.imshow(proc_window, img8)

                # Store corrected cube
                try:
                    hdf5.queue.put_nowait((frame_time, data_cube_corr.copy()))
                    num_cubes_stored += 1
                except Exception:
                    logger.debug('Status:Storage Queue is full!')

            while not hdf5.log.empty():
                (level, msg) = hdf5.log.get_nowait()
                logger.log(level, 'Status:%s', msg)

            if (current_time - last_stats) >= measure_time:
                logger.info('Status:cubes generated/s: %s', num_cubes_generated / measure_time)
                logger.info('Status:cubes queued/s: %s', num_cubes_stored / measure_time)
                logger.debug('Status:frames displayed/s: %s', num_frames_displayed / measure_time)
                num_cubes_generated = 0
                num_cubes_stored = 0
                num_frames_displayed = 0
                last_stats = current_time

            if (current_time - last_display) >= display_interval:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fps = getattr(camera, 'measured_fps', float('nan'))
                cv2.putText(
                    display_frame,
                    f'Capture FPS:{fps} [Hz]',
                    textLocation0,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord('q'):
                    break

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
