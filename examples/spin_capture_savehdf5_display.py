##########################################################################
# Spinnaker capture + display + store cubes to HDF5
#
# Notes
# - Collects 14 frames into a cube and writes the cube to HDF5.
# - Displays frames at a throttled rate to keep the UI responsive.
# - Output goes to %TEMP% on Windows or $TMPDIR (/tmp) on Linux.
##########################################################################

from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _default_output_dir() -> Path:
    if platform.system() == 'Windows':
        return Path(os.environ.get('TEMP', 'C:/temp'))
    return Path(os.environ.get('TMPDIR', '/tmp'))


def main() -> None:

    # Camera configuration file
    from configs.spin_configs import configs

    displayfps = float(configs.get("displayfps", 0) or 0)
    capture_fps = float(configs.get("fps", 0) or 0)
    if displayfps <= 0:
        display_interval = 0.0
    elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
        display_interval = 0.0
    else:
        display_interval = 1.0 / displayfps
    loop_interval = 1.0 / (2.0 * (capture_fps if capture_fps > 0 else 30.0))
    dps_measure_time = 5.0

    depth = 14
    data_cube = None

    window_name = 'Camera'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 480)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('SpinnakerHDF5Display')

    from camera.streamer.h5storageserver import h5Server

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hdf5'
    hdf5 = h5Server(str(output_dir / filename))
    logger.info('Starting Storage Server: %s', output_dir / filename)
    hdf5.start()

    from camera.capture.spincapture import spinCapture

    logger.info('Starting Capture')
    camera = spinCapture(configs)
    camera.start()

    frame_idx = 0
    num_cubes_sent = 0
    num_cubes_generated = 0
    last_stats = time.perf_counter()
    last_display = last_stats
    num_frames_displayed = 0

    frame = None

    stop = False
    try:
        while not stop and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            current_time = time.perf_counter()

            # Pull frame (non-blocking)
            if camera.buffer and camera.buffer.avail > 0:
                frame, frame_time = camera.buffer.pull(copy=False)
            else:
                frame = None
                frame_time = None

            # Drain logs
            try:
                while not camera.log.empty():
                    level, msg = camera.log.get_nowait()
                    logger.log(level, "%s", msg)
                while not hdf5.log.empty():
                    level, msg = hdf5.log.get_nowait()
                    logger.log(level, "%s", msg)
            except Exception:
                pass

            if frame is not None:
                if data_cube is None:
                    data_cube = np.zeros((depth, *frame.shape), dtype=frame.dtype)
                data_cube[frame_idx, ...] = frame
                frame_idx += 1

            if frame_idx >= depth and data_cube is not None:
                frame_idx = 0
                num_cubes_generated += 1

                try:
                    hdf5.queue.put_nowait((frame_time, data_cube.copy()))
                    num_cubes_sent += 1
                except Exception:
                    logger.debug('Status:Storage Queue is full!')

            if frame is not None and (current_time - last_display) >= display_interval:
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Capture FPS:{getattr(camera, "measured_fps", float("nan")):.1f} [Hz]', textLocation0, font, 1, fontColor, lineType)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord('q'):
                    stop = True
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                    stop = True

                last_display = current_time
                num_frames_displayed += 1

            if (current_time - last_stats) >= dps_measure_time:
                logger.debug('Status:cubes generated per second:%s', num_cubes_generated / dps_measure_time)
                logger.debug('Status:cubes sent per second:%s', num_cubes_sent / dps_measure_time)
                logger.debug('Status:Frames displayed per second:%s', num_frames_displayed / dps_measure_time)
                num_cubes_generated = 0
                num_cubes_sent = 0
                num_frames_displayed = 0
                last_stats = current_time

            remaining = loop_interval - (time.perf_counter() - current_time)
            if remaining > 0:
                time.sleep(remaining)

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
            if hasattr(camera, "close"):
                camera.close()
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
