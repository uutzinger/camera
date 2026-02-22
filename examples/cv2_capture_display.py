##########################################################################
# OpenCV (cv2Capture) display + capture example
#
# - Uses the cv2Capture wrapper (FrameBuffer-based)
# - Displays frames via OpenCV (no analysis)
# - Throttles display rate in THIS demo (consumer-side)
#
# Notes:
# - On Jetson Nano (aarch64) this will use nanoCapture (Queue-based legacy)
# - On other platforms it uses cv2Capture (FrameBuffer-based)
##########################################################################

from __future__ import annotations

import logging
import os
import platform
import sys
import time

if sys.platform.startswith("win"):
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

from camera.utils import probeCameras


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
        return 0.0
    if capture_fps > 0 and display_fps >= 0.8 * capture_fps:
        return 0.0
    return 1.0 / display_fps


def select_camera_index(width_sig=640, height_sig=480, fourcc_sig="\x16\x00\x00\x00") -> int:
    """Best-effort probe to pick a camera that matches a signature."""
    camera_index = 0
    try:
        camprops = probeCameras(5)
    except Exception:
        return camera_index

    score = 0
    for i in range(len(camprops)):
        try:
            found_fourcc = 1 if camprops[i].get("fourcc") == fourcc_sig else 0
        except Exception:
            found_fourcc = 0
        try:
            found_width = 1 if camprops[i].get("width") == width_sig else 0
        except Exception:
            found_width = 0
        try:
            found_height = 1 if camprops[i].get("height") == height_sig else 0
        except Exception:
            found_height = 0
        tmp = found_fourcc + found_width + found_height
        if tmp > score:
            score = tmp
            camera_index = i
    return camera_index


def main() -> None:
    setup_opencv()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cv2_capture_display")

    # Pick a camera index (best-effort)
    camera_index = select_camera_index()

    # Import or define camera configuration below
    # Make sure you have configs folder in the folder of your main program
    # -Eluktronics Max-15 internal camera
    from configs.eluk_configs import configs as configs
    # -Dell Inspiron 15 internal camera
    # from configs.dell_internal_configs  import configs as configs
    # -Generic webcam
    # from configs.generic_1080p import configs as configs
    # -Nano Jetson IMX219 camera
    # from configs.nano_IMX219_configs  import configs as configs
    # -Raspberry Pi v1 & v2 camera (if using cv2 backend)
    # from configs.raspi_v1module_configs  import configs as configs
    # from configs.raspi_v2module_configs  import configs as configs
    # -ELP MAX15 internal camera
    # from configs.ELP1080p_configs  import configs as configs
    # -FLIR Lepton 3.5
    # from configs.FLIRlepton35 import confgis as configs

    display_interval = display_interval_from_config(configs)
    dps_measure_interval = 5.0

    # Display Window Setup
    window_name = "Camera"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 40)
    textLocation2 = (10, 60)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Create camera interface based on OS
    plat = platform.system()
    if plat == "Linux" and platform.machine() == "aarch64":  # Jetson Nano
        from camera.capture.nanocapture import nanoCapture

        camera = nanoCapture(configs, camera_index)
        use_framebuffer = False
    else:
        from camera.capture.cv2capture import cv2Capture

        camera = cv2Capture(configs, camera_index)
        use_framebuffer = getattr(camera, "buffer", None) is not None

    logger.info("Getting Images")
    camera.start()

    last_display = time.perf_counter()
    last_dps_time = last_display
    measured_dps = 0.0
    num_frames_displayed = 0

    loop_interval = 1.0 / (2.0 * float(configs.get("fps", 30) or 30))

    stop = False
    try:
        while not stop and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            current_time = time.perf_counter()

            # Pull frame
            if use_framebuffer:
                if camera.buffer and camera.buffer.avail > 0:
                    frame, _ts = camera.buffer.pull(copy=False)
                else:
                    frame = None
            else:
                # Legacy Queue-based path (nanoCapture)
                try:
                    _ts, frame = camera.capture.get(timeout=0.25)
                except Exception:
                    frame = None

            # Drain logs
            try:
                while not camera.log.empty():
                    level, msg = camera.log.get_nowait()
                    logger.log(level, "%s", msg)
            except Exception:
                pass

            # Display
            delta_display = current_time - last_display
            if (frame is not None) and (delta_display >= display_interval):
                frame_display = frame.copy()
                cv2.putText(
                    frame_display,
                    f"Capture FPS:{camera.measured_fps:.1f} [Hz]",
                    textLocation0,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )
                cv2.putText(
                    frame_display,
                    f"Display FPS:{measured_dps:.1f} [Hz]",
                    textLocation1,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )
                cv2.putText(
                    frame_display,
                    f"Res:{configs.get('camera_res')}",
                    textLocation2,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )

                cv2.imshow(window_name, frame_display)
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

            loop_remaining_time = loop_interval - (time.perf_counter() - current_time)
            if loop_remaining_time > 0.0:
                time.sleep(loop_remaining_time)

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
