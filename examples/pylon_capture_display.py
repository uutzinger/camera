##########################################################################
# Basler pylon capture + display example
#
# What this script does
# - Captures frames from a Basler camera via `pylonCapture` (pypylon).
# - Displays frames at a throttled rate (`displayfps`) while capturing at the
#   configured camera rate (`fps`).
# - Logs capture and display throughput every ~5 seconds.
#
# Notes
# - Requires Basler pylon runtime and `pypylon`.
# - Uses `pylonCapture.measured_fps` for measured capture rate.
##########################################################################

from __future__ import annotations

import logging
import time

import cv2


def main() -> None:
    from configs.pylon_configs import configs

    displayfps = float(configs.get("displayfps", 0) or 0)
    capture_fps = float(configs.get("fps", 0) or 0)
    if displayfps <= 0:
        display_interval = 0.0
    elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
        display_interval = 0.0
    else:
        display_interval = 1.0 / displayfps
    loop_interval = 1.0 / (2.0 * (capture_fps if capture_fps > 0 else 30.0))

    window_name = "Camera"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_location0 = (10, 20)
    text_location1 = (10, 60)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Pylon")

    from camera.capture.pyloncapture import pylonCapture

    logger.info("Starting Capture")
    camera = pylonCapture(configs)
    camera.start()

    stats_interval = 5.0
    last_display = time.perf_counter()
    last_stats = last_display
    measured_dps = 0.0
    num_frames_displayed = 0

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            if camera.buffer and camera.buffer.avail > 0:
                frame, _ts = camera.buffer.pull(copy=False)
            else:
                frame = None

            try:
                while not camera.log.empty():
                    level, msg = camera.log.get_nowait()
                    logger.log(level, "%s", msg)
            except Exception:
                pass

            if frame is not None and (current_time - last_display) >= display_interval:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Capture FPS:{camera.measured_fps:.1f} [Hz]", text_location0, font, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Display FPS:{measured_dps:.1f} [Hz]", text_location1, font, 1, (255, 255, 255), 2)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord("q"):
                    stop = True
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                        stop = True
                except Exception:
                    stop = True

                last_display = current_time
                num_frames_displayed += 1

            if (current_time - last_stats) >= stats_interval:
                measured_dps = num_frames_displayed / stats_interval
                logger.debug("Status:Frames displayed per second:%s", measured_dps)
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
