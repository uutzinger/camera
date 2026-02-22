##########################################################################
# Spinnaker capture + display example
#
# What this script does
# - Captures frames from a FLIR Spinnaker camera via `spinCapture` (PySpin).
# - Displays frames at a throttled rate (`displayfps`) while capturing at full
#   camera rate (`fps`). This keeps UI responsive even at very high FPS.
# - Logs capture throughput and display throughput every ~5 seconds.
#
# Notes
# - Requires the FLIR Spinnaker SDK / PySpin to be installed.
# - Uses `spinCapture.measured_fps` for the measured capture rate.
##########################################################################
# Results
#    Capture FPS: 524.2
#    Display FPS: 32.2
#    CPU Usage: 6-7%
##########################################################################

from __future__ import annotations

import logging
import time

import cv2


def main() -> None:

    configs = {
        'camera_res': (720, 540),
        'exposure': 1750,          # microseconds, -1 = autoexposure
        'autoexposure': 0,         # 0,1
        'fps': 500,
        'binning': (1, 1),
        'offset': (0, 0),
        'adc': 8,                  # 8,10,12,14 bit
        'trigout': 2,              # -1 no trigger output
        'ttlinv': True,
        'trigin': -1,
        'output_res': (-1, -1),
        'flip': 0,
        'displayfps': 50,
    }

    displayfps = float(configs.get("displayfps", 0) or 0)
    capture_fps = float(configs.get("fps", 0) or 0)
    if displayfps <= 0:
        display_interval = 0.0
    elif capture_fps > 0 and displayfps >= 0.8 * capture_fps:
        display_interval = 0.0
    else:
        display_interval = 1.0 / displayfps
    loop_interval = 1.0 / (2.0 * (capture_fps if capture_fps > 0 else 30.0))

    window_name = 'Camera'
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 60)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('Spinnaker')

    from camera.capture.spincapture import spinCapture

    logger.info('Starting Capture')
    camera = spinCapture(configs)
    camera.start()

    dps_measure_time = 5.0
    last_display = time.perf_counter()
    last_time = last_display
    measured_dps = 0.0
    num_frames_displayed = 0

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            # Pull latest frame from framebuffer (non-blocking)
            if camera.buffer and camera.buffer.avail > 0:
                frame, _ts = camera.buffer.pull(copy=False)
            else:
                frame = None

            # Drain logs
            try:
                while not camera.log.empty():
                    level, msg = camera.log.get_nowait()
                    logger.log(level, "%s", msg)
            except Exception:
                pass

            if frame is not None and (current_time - last_display) > display_interval:
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Capture FPS:{camera.measured_fps:.1f} [Hz]', textLocation0, font, fontScale, fontColor, lineType)
                cv2.putText(display_frame, f'Display FPS:{measured_dps:.1f} [Hz]', textLocation1, font, fontScale, fontColor, lineType)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1)
                if (key & 0xFF) == ord('q'):
                    stop = True
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                        stop = True
                except Exception:
                    stop = True

                last_display = current_time
                num_frames_displayed += 1

            if (current_time - last_time) >= dps_measure_time:
                measured_dps = num_frames_displayed / dps_measure_time
                logger.debug('Status:Frames displayed per second:%s', measured_dps)
                num_frames_displayed = 0
                last_time = current_time

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


if __name__ == '__main__':
    main()
