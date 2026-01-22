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
        'fps'             : 60,             # requested capture frame rate
        'autoexposure'    : -1,             # -1 leave unchanged (match picamera2_direct)
        # 'aemeteringmode'  : -1,             # -1 leave unchanged
        'autowb'          : -1,             # -1 leave unchanged
        #'awbmode'         : -1,             # -1 leave unchanged
        # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
        # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
        'format'          : 'BGR888',
        "stream_policy"   : "default",      # default maps to maximize_fps_no_crop
        'low_latency'     : False,          # match picamera2_direct defaults
        'buffersize'      : 4,              # wrapper buffer depth (not in direct script)
        'buffer_overwrite': True,           # overwrite old frames if buffer full
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
    last_fps_log_time = time.perf_counter()
    try:
        while not stop:

            # Drain all pending frames so the consumer doesn't fall behind.
            # Use copy=False to avoid extra memcpy overhead in the consumer.
            if camera.buffer.avail > 0:
                # Use copy=False to avoid extra memcpy; we copy only when displaying.
                frame, _frame_time = camera.buffer.pull(copy=False)
            else:
                frame = None
                time.sleep(0.001)


            # display log
            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "{}".format(msg))

            # display fps periodically
            now = time.perf_counter()
            if now- last_fps_log_time > 5.0:
                fps = camera.measured_fps
                if fps is not None:
                    logger.log(logging.INFO, "FPS: %.1f", fps)
                last_fps_log_time = now

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
            camera.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
