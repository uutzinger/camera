##########################################################################
# Spinnaker capture + store cubes to HDF5 (no display)
#
# Notes
# - Collects 14 frames into a cube and writes each cube as a dataset to HDF5.
# - Requires the FLIR Spinnaker SDK / PySpin to be installed.
# - Output goes to %TEMP% on Windows or $TMPDIR (/tmp) on Linux.
##########################################################################

from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _default_output_dir() -> Path:
    if platform.system() == 'Windows':
        return Path(os.environ.get('TEMP', 'C:/temp'))
    return Path(os.environ.get('TMPDIR', '/tmp'))

def main() -> None:

    depth = 14

    # Camera configuration file
    from configs.spin_configs import configs
    capture_fps = float(configs.get("fps", 0) or 0)
    loop_interval = 1.0 / (2.0 * (capture_fps if capture_fps > 0 else 30.0))

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('SpinnakerHDF5')

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

    data_cube = None
    frame_idx = 0
    num_cubes_sent = 0
    num_cubes_received = 0
    last_time = time.perf_counter()

    try:
        while True:
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

            if frame is None:
                remaining = loop_interval - (time.perf_counter() - current_time)
                if remaining > 0:
                    time.sleep(remaining)
                continue

            if data_cube is None:
                data_cube = np.zeros((depth, *frame.shape), dtype=frame.dtype)

            data_cube[frame_idx, ...] = frame
            frame_idx += 1

            if frame_idx >= depth:
                frame_idx = 0
                num_cubes_received += 1

                try:
                    hdf5.queue.put_nowait((frame_time, data_cube.copy()))
                    num_cubes_sent += 1
                except Exception:
                    logger.warning('Status:Storage Queue is full!')

            if (current_time - last_time) >= 5.0:
                logger.info('Status:Cubes received per second:%s', num_cubes_received / 5.0)
                logger.info('Status:Cubes sent per second:%s', num_cubes_sent / 5.0)
                last_time = current_time
                num_cubes_received = 0
                num_cubes_sent = 0

            remaining = loop_interval - (time.perf_counter() - current_time)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
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


if __name__ == '__main__':
    main()
