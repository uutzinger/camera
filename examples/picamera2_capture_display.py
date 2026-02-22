##########################################################################
# Picamera2 Capture + Display (Raspberry Pi)
#
# - Uses the Picamera2 threaded wrapper
# - Displays frames via OpenCV (no analysis)
# - Throttles display rate in THIS demo (consumer-side)
##########################################################################

from __future__ import annotations

import logging
import os
import time
import cv2

# Urs' camera wrapper
from camera.capture.picamera2capture import piCamera2Capture

def setup_opencv() -> None:
    """Small performance tweaks for OpenCV on low-power devices."""
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass

def display_interval_from_config(configs: dict) -> float:
    display_fps = float(configs.get("displayfps", 0) or 0)
    capture_fps = float(configs.get("fps", 0) or 0)

    if display_fps <= 0:
        return 0.0  # no throttling
    if capture_fps > 0 and display_fps >= 0.8 * capture_fps:
        return 0.0  # close to capture fps so no throttling
    return 1.0 / display_fps  # throttled display


def main() -> None:
    setup_opencv()

    # Setting up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PiCamera2 Capture")

    # Silence Picamera2 logging
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")  # 3=ERROR, 2=WARNING, 1=INFO, 0=DEBUG

    camera_index = 0 # Take first camera

    configs = {
        ################################################################
        # Picamera2 capture configuration
        ################################################################
        # Capture mode:
        #   'main' -> full-FOV processed stream (BGR/YUV), scaled to 'camera_res' (libcamera scales)
        #   'raw'  -> high-FPS raw sensor window (exact sensor mode only), cropped FOV
        'mode'            : 'main',
        'camera_res'      : (640, 480),     # requested main stream size (w, h)
        'exposure'        : 0,              # microseconds, 0/-1 for auto
        'fps'             : 60,             # requested capture frame rate
        'autoexposure'    : -1,             # -1 leave unchanged, 0 AE off, 1 AE on
        #'aemeteringmode'  : -1,             # int or 'center'|'spot'|'matrix' 
        'autowb'          : -1,             # -1 leave unchanged, 0 AWB off, 1 AWB on
        #'awbmode'         : -1,             # int or friendly string 'auto'|'incandescent'|'fluorescent'|'warm-fluorescent'|'daylight'|'cloudy-daylight'|'twilight'|'shade'
        # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
        # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
        'format'          : 'BGR888',
        "stream_policy"   : "default",      # 'default', 'maximize_fps_no_crop', 'maximize_fps_with_crop', 'maximize_fov'
        'low_latency'     : False,          # low_latency=True prefers size-1 buffer (latest frame)
        'buffersize'      : 4,              # capture queue size override (wrapper-level)
        'buffer_overwrite': True,           # overwrite old frames if buffer full
        'output_res'      : (-1, -1),       # (-1,-1): output == input; else libcamera scales main
        'flip'            : 0,              # 0=norotation 
        'displayfps'      : 30              # frame rate for display server
    }

    display_interval = display_interval_from_config(configs) # desired display interval
    dps_measure_interval = 5.0 # measure displayed frames per second over this interval

    # Display Window Setup
    window_name = "Camera"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 40)
    textLocation2 = (10, 60)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # Create window for display

    # Camera Setup
    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not camera.cam_open:
        raise RuntimeError("PiCamera2 camera failed to open")

    # Logging Setup
    logger.log(logging.INFO, "Getting Images")
    # Log configuration
    logger.log(
        logging.INFO,
        "Config: mode=%s format=%s camera_res=%s output_res=%s",
        configs.get("mode"),
        configs.get("format"),
        configs.get("camera_res"),
        configs.get("output_res"),
    )
    # Log stream options
    camera.log_stream_options() # Optional show suggested main options or raw sensor modes
    # Start Camera
    camera.start()
    # Log actual camera configuration and controls after camera started
    camera.log_camera_config_and_controls()

    # Initialize variables for main loop
    last_display = time.perf_counter()
    last_dps_time = last_display
    measured_dps = 0.0
    num_frames_displayed = 0
    logged_camera_controls = False
    stop = False
    loop_interval = 1./(2.*configs.get("fps", 30))

    ####################################################################################
    # Main loop: Capture + Display
    ####################################################################################
    try:
        while not stop:
            current_time = time.perf_counter()

            # Pull latest available frame.
            if camera.buffer.avail > 0:
                # Use copy=False to avoid extra memcpy; we copy only when displaying.
                frame, _frame_time = camera.buffer.pull(copy=False)
            else:
                frame = None

            # Display log information
            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "{}".format(msg))

            # Display frame (throttled)
            delta_display = current_time - last_display
            if (frame is not None) and (delta_display >= display_interval):
                frame_display = frame.copy()
                # Overlay text information
                cv2.putText(frame_display, "Capture FPS:{:.1f} [Hz]".format(camera.measured_fps),
                    textLocation0, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, "Display FPS:{:.1f} [Hz]".format(measured_dps),
                    textLocation1, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, f"Mode:{configs.get('mode')}",
                    textLocation2, font, fontScale, fontColor, lineType,
                )
                # Display frame
                cv2.imshow(window_name, frame_display)
                # Check for 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True

                last_display = current_time
                num_frames_displayed += 1

            # Update display FPS measurement
            delta_dps = current_time - last_dps_time
            if delta_dps >= dps_measure_interval:
                measured_dps = num_frames_displayed / delta_dps
                num_frames_displayed = 0
                last_dps_time = current_time

            # Give some time to other threads
            loop_remaining_time = loop_interval - (time.perf_counter() - current_time)
            if loop_remaining_time > 0.:
                time.sleep(loop_remaining_time)

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
