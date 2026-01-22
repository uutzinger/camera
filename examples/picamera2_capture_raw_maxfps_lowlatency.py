##########################################################################
# Picamera2 RAW capture (maximize FPS) + OpenCV display (Raspberry Pi)
#
# Dedicated Picamera2/libcamera demo:
# - Uses the current Picamera2 threaded wrapper
# - RAW Bayer stream, prefers highest-FPS sensor mode
# - low_latency=True for minimal buffering (size-1 buffer)
# - Throttles display rate (consumer-side)
##########################################################################

from __future__ import annotations

import logging
import os
import time

import cv2


def main() -> None:
    # Optimize OpenCV performance on small CPUs
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PiCamera2 RAW MaxFPS")

    # Silence Picamera2 / libcamera logs; keep only this script's logging output
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")

    camera_index = 0

    configs = {
        ################################################################
        # Picamera2 RAW capture configuration
        #
        # List camera properties with:
        #     examples/list_Picamera2Properties.py
        ################################################################
        # Capture mode:
        #   'main' -> full-FOV processed stream (BGR/YUV)
        #   'raw'  -> RAW Bayer sensor mode (exact sensor mode only), may crop FOV
        'mode'            : 'raw',
        'camera_res'      : (640, 480),     # hint resolution (w, h)
        'output_res'      : (-1, -1),       # RAW: keep sensor mode size; avoid CPU resize
        'fps'             : 120,            # requested capture frame rate (sensor modes may limit)
        'exposure'        : 0,              # microseconds, 0/-1 for auto
        'autoexposure'    : 1,              # -1 leave unchanged, 0 AE off, 1 AE on
        'aemeteringmode'  : 'center',       # int or 'center'|'spot'|'matrix'
        'autowb'          : 1,              # -1 leave unchanged, 0 AWB off, 1 AWB on
        'awbmode'         : 'auto',         # int or friendly string
        # RAW stream formats: SRGGB8, SGRBG10_CSI2P, ... (see properties script)
        'raw_format'      : 'SGRBG',        # Bayer pattern/format hint
        'stream_policy'   : 'maximize_fps_no_crop', # 'default', 'maximize_fps_no_crop', 'maximize_fps_with_crop', 'maximize_fov'
        'low_latency'     : True,           # size-1 queue (latest frame)
        'flip'            : 0,              # 0=norotation
        'displayfps'      : 15,             # consumer-side display throttle (0 disables)
    }

    displayfps  = float(configs.get('displayfps', 0) or 0)
    capture_fps = float(configs.get('fps', 0) or 0)
    if displayfps <= 0:
        display_interval = 0.0 # no throttling
    elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
        display_interval = 0.0 # close to capture fps so no throttling
    else:
        display_interval = 1.0 / displayfps # throttled display

    window_name = "PiCamera2 RAW MaxFPS"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 40)
    textLocation2 = (10, 60)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    from camera.capture.picamera2capture import piCamera2Capture

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not camera.cam_open:
        raise RuntimeError("PiCamera2 camera failed to open in RAW mode")

    logger.info("Getting RAW images (mode=raw, stream_policy=maximize_fps_no_crop, low_latency=True)")
    logger.info(
        'Config: mode=%s raw_format=%s camera_res=%s output_res=%s fps=%s',
        configs.get('mode'),
        configs.get('raw_format'),
        configs.get('camera_res'),
        configs.get('output_res'),
        configs.get('fps'),
    )

    # One-time capability dump (RAW sensor modes, plus selection summary)
    camera.log_stream_options()

    camera.start()

    last_display = time.perf_counter()

    stop = False
    frame = None
    try:
        while not stop:
            current_time = time.perf_counter()

            if camera.buffer.avail > 0:
                # Grab the latest frame without extra copies.
                frame, _frame_time = camera.buffer.pull(copy=False)

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "%s", msg)

            if (frame is not None) and ((current_time - last_display) >= display_interval):
                frame_display = frame.copy()
                cv2.putText(frame_display, f"Capture FPS:{camera.measured_fps:.1f} [Hz]",
                    textLocation0, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, f"Display FPS:{displayfps:.1f} [Hz]",
                    textLocation1, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, "Mode:raw Policy:maximize_fps_no_crop Low-latency:True",
                    textLocation2, font, fontScale, fontColor, lineType,
                )
                cv2.imshow(window_name, frame_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop = True
                # if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                #    stop = True

                last_display = current_time

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
            camera.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
