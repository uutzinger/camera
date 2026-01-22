##########################################################################
# Picamera2 synthetic capture (headless)
#
# Dedicated Picamera2/libcamera demo without any OpenCV display.
# - Uses the picamera2 threaded wrapper
# - Logs capture FPS periodically for benchmarking
##########################################################################

from __future__ import annotations

import logging
import os
import time

from camera.capture.picamera2capture import piCamera2Capture


def main() -> None:
    # Setting up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PiCamera2 Synthetic Capture")

    # Silence Picamera2 / libcamera logs; keep only this script's logging output
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")  # 3=ERROR, 2=WARNING, 1=INFO, 0=DEBUG

    camera_index = 0

    configs = {
        ################################################################
        # Picamera2 capture configuration
        #
        # List camera properties with:
        #     examples/list_Picamera2Properties.py
        ################################################################
        # Capture mode:
        #   'main' -> full-FOV processed stream (BGR/YUV), scaled to 'camera_res' (libcamera scales)
        #   'raw'  -> high-FPS raw sensor window (exact sensor mode only), cropped FOV
        'mode'            : 'main',
        'camera_res'      : (640, 480),     # requested main stream size (w, h)
        'exposure'        : 0,              # microseconds, 0/-1 for auto
        'fps'             : 60,             # requested capture frame rate
        'autoexposure'    : 1,              # -1 leave unchanged, 0 AE off, 1 AE on
        'aemeteringmode'  : 'center',       # int or 'center'|'spot'|'matrix'
        'autowb'          : 1,              # -1 leave unchanged, 0 AWB off, 1 AE on
        'awbmode'         : 'auto',         # int or friendly string
        # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
        # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
        'format'          : 'BGR3',
        "stream_policy"   : "default",      # 'default', 'maximize_fps_no_crop', 'maximize_fps_with_crop', 'maximize_fov'
        'low_latency'     : True,           # low_latency=True prefers size-1 buffer (latest frame)
        'buffersize'      : 4,              # capture queue size override (wrapper-level)
        'output_res'      : (-1, -1),       # (-1,-1): output == input; else libcamera scales main
        'flip'            : 0,              # 0=norotation 
        'displayfps'      : 0,             # historical config entry (ignored here)
        'test_pattern'    : 'gradient',     # enable synthetic frames (bypass Picamera2/libcamera)
    }

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not camera.cam_open:
        raise RuntimeError("PiCamera2 camera failed to open")

    logger.info("PiCamera2 synthetic capture starting")
    logger.info(
        "Config: mode=%s format=%s camera_res=%s output_res=%s",
        configs.get('mode'),
        configs.get('format'),
        configs.get('camera_res'),
        configs.get('output_res'),
    )

    # Optional show suggested main options or raw sensor modes
    camera.log_stream_options()

    camera.start()

    report_interval = 5.0
    last_report = time.perf_counter()
    frames_since_report = 0

    try:
        while True:
            if camera.buffer.avail > 0:
                camera.buffer.pull(copy=False)
                frames_since_report += 1

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "%s", msg)

            now = time.perf_counter()
            if (now - last_report) >= report_interval:
                logger.info(
                    "Capture FPS (PiCamera2 measured): %.1f Hz", float(camera.measured_fps)
                )
                logger.info(
                    "Frames drained: %d in %.2f s", frames_since_report, now - last_report
                )
                frames_since_report = 0
                last_report = now

            time.sleep(0.002)
    except KeyboardInterrupt:
        logger.info("Stopping synthetic capture (Ctrl+C)")
    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
