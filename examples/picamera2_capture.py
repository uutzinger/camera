##########################################################################
# Picamera2 capture + OpenCV display (Raspberry Pi)
#
# Dedicated Picamera2/libcamera demo:
# - Uses the Picamera2 threaded wrapper
# - Throttles display rate
##########################################################################

from __future__ import annotations

import logging
import os
import time

def main() -> None:

    # Setting up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PiCamera2 Capture")

    # Also silence libcamera C++ logs via environment (must be set before libcamera loads)
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
        'fps'             : 0,              # 0: do not request; matches picamera2_direct defaults
        'autoexposure'    : 1,              # -1 leave unchanged, 0 AE off, 1 AE on
        'aemeteringmode'  : 'center',       # int or 'center'|'spot'|'matrix'
        'autowb'          : 1,              # -1 leave unchanged, 0 AWB off, 1 AWB on
        'awbmode'         : 'auto',         # int or friendly string
        # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
        # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
        'format'          : 'BGR3',
        "stream_policy"   : "maximize_fps", # match picamera2_direct defaults
        'low_latency'     : False,          # match picamera2_direct defaults
        'buffersize'      : 4,              # wrapper buffer depth (not in direct script)
        'output_res'      : (-1, -1),       # (-1,-1): output == input; else libcamera scales main
        'flip'            : 0,              # 0=norotation 
    }

    # Camera
    from camera.capture.picamera2capture import piCamera2Capture

    camera = piCamera2Capture(configs, camera_num=camera_index)
    # open_cam() is called in __init__, so no need to call it again
    if not camera.cam_open:
        raise RuntimeError("PiCamera2 camera failed to open")

    logger.log(logging.INFO, "Getting Images")
    logger.log(
        logging.INFO,
        "Config: mode=%s format=%s camera_res=%s output_res=%s",
        configs.get("mode"),
        configs.get("format"),
        configs.get("camera_res"),
        configs.get("output_res"),
    )

    # Optional show suggested main options or raw sensor modes
    camera.log_stream_options()

    camera.start()
    try:
        camera.log_camera_config_and_controls()
    except Exception:
        pass

    stop = False
    frame = None
    try:
        while not stop:

            # Drain all pending frames so the consumer doesn't fall behind.
            # Use copy=False to avoid extra memcpy overhead in the consumer.
            while camera.buffer.avail > 0:
                frame, _ts_ms = camera.buffer.pull(copy=False)

            # display log
            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "{}".format(msg))

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
            camera.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
