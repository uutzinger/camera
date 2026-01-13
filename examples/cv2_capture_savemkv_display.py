##########################################################################
# Capture + display + store MKV (cv2Capture)
#
# Notes
# - Uses mkvServer (threaded VideoWriter) to write frames to disk.
# - Optionally scans connected cameras and picks the one matching a simple
#   signature (width/height/fourcc) via camera.utils.probeCameras.
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

from camera.utils import probeCameras


def _default_output_dir() -> Path:
    if platform.system() == 'Windows':
        return Path(os.environ.get('TEMP', 'C:/temp'))
    return Path(os.environ.get('TMPDIR', '/tmp'))


def _pick_camera_index(max_cameras: int = 10) -> int:
    """Pick the best matching camera index using a simple signature."""

    # Signature examples (legacy heuristic)
    widthSig = 640
    heightSig = 480
    fourccSig = "\x16\x00\x00\x00"  # e.g. YUYV-like on some systems

    camera_index = 0

    camprops = probeCameras(max_cameras)
    score = 0
    for i in range(len(camprops)):
        found_fourcc = 1 if camprops[i].get('fourcc') == fourccSig else 0
        found_width = 1 if camprops[i].get('width') == widthSig else 0
        found_height = 1 if camprops[i].get('height') == heightSig else 0
        tmp = found_fourcc + found_width + found_height
        if tmp > score:
            score = tmp
            camera_index = i

    return camera_index


def main() -> None:

    # Disable HW transforms on Windows MSMF to reduce surprise rotations
    if platform.system() == 'Windows':
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

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

    camera_index = _pick_camera_index(10)

    window_name = 'Camera'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 60)
    textLocation2 = (10, 100)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('cv2_capture_savemkv_display')

    from camera.streamer.mkvstorageserver import mkvServer

    now = datetime.now()
    fps = configs['fps']
    size = configs['camera_res']

    output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = now.strftime('%Y%m%d%H%M%S') + '.mkv'
    out_path = output_dir / filename

    mkv = mkvServer(str(out_path), fps, size)
    logger.info('Starting MKV storage: %s', out_path)
    mkv.start()

    logger.info('Starting Capture (index=%s)', camera_index)

    plat = platform.system()
    if plat == 'Linux' and platform.machine() == 'aarch64':
        from camera.capture.nanocapture import nanoCapture

        camera = nanoCapture(configs, camera_index)
    else:
        from camera.capture.cv2capture import cv2Capture

        camera = cv2Capture(configs, camera_index)

    camera.start()

    num_frames_sent = 0
    last_stats = time.perf_counter()
    last_display = time.perf_counter()
    num_frames_displayed = 0
    measured_dps = 0.0

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            try:
                (frame_time, frame) = camera.capture.get(block=True, timeout=1.0)
            except Empty:
                frame = None

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, '%s', msg)

            if frame is not None:
                try:
                    mkv.queue.put_nowait((frame_time, frame))
                    num_frames_sent += 1
                except Exception:
                    logger.warning('Status:Storage Queue is full!')

            while not mkv.log.empty():
                (level, msg) = mkv.log.get_nowait()
                logger.log(level, '%s', msg)

            if (current_time - last_stats) >= 5.0:
                measured_fps_sent = num_frames_sent / 5.0
                measured_dps = num_frames_displayed / 5.0
                logger.info('Status:frames sent to storage per second:%s', measured_fps_sent)
                logger.info('Status:frames displayed per second:%s', measured_dps)
                num_frames_sent = 0
                num_frames_displayed = 0
                last_stats = current_time

            if frame is not None and (current_time - last_display) >= (0.8 * display_interval):
                frame_display = frame.copy()
                fps_cap = getattr(camera, 'measured_fps', float('nan'))

                cv2.putText(frame_display, f'Capture FPS:{fps_cap} [Hz]', textLocation0, font, fontScale, fontColor, lineType)
                cv2.putText(frame_display, f'Display FPS:{measured_dps} [Hz]', textLocation1, font, fontScale, fontColor, lineType)
                cv2.putText(frame_display, f'Storage FPS:{getattr(mkv, "measured_cps", float("nan"))} [Hz]', textLocation2, font, fontScale, fontColor, lineType)

                cv2.imshow(window_name, frame_display)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord('q'):
                    stop = True

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                    stop = True

                last_display = current_time
                num_frames_displayed += 1

    finally:
        try:
            camera.stop()
        except Exception:
            pass
        try:
            mkv.stop()
            mkv.join(timeout=2.0)
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
