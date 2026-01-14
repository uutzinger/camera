##########################################################################
# Picamera2 wrapper + OpenCV display (SYNTHETIC frames)
#
# This is the synthetic counterpart to:
#   examples/picamera2_capture_display.py
#
# It uses the same wrapper camera loop + FrameBuffer pipeline, but the core
# generates frames internally (no Picamera2/libcamera I/O).
#
# Use this to isolate wrapper overhead (thread loop, conversion, ring buffer).
##########################################################################

from __future__ import annotations

import logging
import os
import time

import cv2


def main() -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PiCamera2 Synthetic Capture")

    # Keep libcamera quiet (even though synthetic mode does not load it).
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")

    configs = {
        # Same shape/behavior as the real demo, but with synthetic frames
        "mode": "main",
        "camera_res": (640, 480),
        "exposure": 0,
        # Stress-test: request high FPS. 0 = unpaced (as fast as possible).
        "fps": 1000,
        "autoexposure": 1,
        "aemeteringmode": "center",
        "autowb": 1,
        "awbmode": "auto",
        # Choose a main format you want to test conversion for.
        # BGR3 -> BGR888 (fast), XRGB8888 / XBGR8888 exercises 4->3 conversion.
        "format": "BGR3",
        "stream_policy": "default",
        "low_latency": True,
        "buffersize": 4,
        "output_res": (-1, -1),
        "flip": 0,
        "displayfps": 30,
        # Enables synthetic generation in PiCamera2Core
        "test_pattern": "gradient",  # gradient|checker|noise|static
    }

    displayfps = float(configs.get("displayfps", 0) or 0)
    capture_fps = float(configs.get("fps", 0) or 0)

    if displayfps <= 0:
        display_interval = 0.0
    elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
        display_interval = 0.0
    else:
        display_interval = 1.0 / displayfps

    window_name = "Camera (synthetic)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text0 = (10, 20)
    text1 = (10, 40)
    text2 = (10, 60)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    from camera.capture.picamera2capture import piCamera2Capture

    camera = piCamera2Capture(configs, camera_num=0)
    if not camera.open_cam():
        raise RuntimeError("Synthetic camera failed to open")

    logger.info("Synthetic capture started")
    camera.start()

    last_display = time.perf_counter()

    stop = False
    try:
        while not stop:
            now = time.perf_counter()

            frame = None
            if getattr(camera, "buffer", None) is not None and camera.buffer.avail() > 0:
                frame, _ts = camera.buffer.pull(copy=False)

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "{}".format(msg))

            if (frame is not None) and ((now - last_display) >= display_interval):
                frame_display = frame.copy()
                cv2.putText(frame_display, f"Capture FPS:{camera.measured_fps:.1f} [Hz]", text0, font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_display, f"Display target:{displayfps:.1f} [Hz]", text1, font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_display, f"Synthetic:{configs.get('test_pattern')}", text2, font, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, frame_display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                last_display = now

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
            camera.close_cam()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
