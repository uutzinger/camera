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

    # Silence Picamera2
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
        'autowb'          : 1,              # -1 leave unchanged, 0 AWB off, 1 AWB on
        'awbmode'         : 'auto',         # int or friendly string
        # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
        # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
        'format'          : 'BGR3',
        "stream_policy"   : "maximize_fps", # 'maximize_fov', 'maximize_fps', 'default'
        'low_latency'     : False,          # low_latency=True prefers size-1 buffer (latest frame)
        'buffersize'      : 4,              # capture queue size override (wrapper-level)
        'output_res'      : (-1, -1),       # (-1,-1): output == input; else libcamera scales main
        'flip'            : 0,              # 0=norotation 
        'displayfps'      : 30              # frame rate for display server
    }

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

    # Camera Setup

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not camera.cam_open:
        raise RuntimeError("PiCamera2 camera failed to open")

    # Logging Setup

    logger.log(logging.INFO, "Getting Images")
    logger.log(
        logging.INFO,
        "Config: mode=%s format=%s camera_res=%s output_res=%s",
        configs.get("mode"),
        configs.get("format"),
        configs.get("camera_res"),
        configs.get("output_res"),
    )
    camera.log_stream_options() # Optional show suggested main options or raw sensor modes
    camera.start()
    camera.log_camera_config_and_controls()

    # Initialize variables for main loop

    last_display = time.perf_counter()
    last_dps_time = last_display
    measured_dps = 0.0
    num_frames_displayed = 0
    logged_camera_controls = False

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            # Pull latest available frame.
            if camera.buffer.avail > 0:
                # Use copy=False to avoid extra memcpy; we copy only when displaying.
                frame, _frame_time = camera.buffer.pull(copy=False)
            else:
                frame = None

            # Display log
            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "{}".format(msg))

            # Display
            delta_display = current_time - last_display
            if (frame is not None) and (delta_display >= display_interval):
                frame_display = frame.copy()
                if not logged_camera_controls:
                    try:
                        fd = camera.get_control("FrameDuration")
                        fdl = camera.get_control("FrameDurationLimits")
                        sc = camera.get_control("ScalerCrop")
                        logger.log(
                            logging.INFO,
                            "Camera controls: FrameDuration=%s FrameDurationLimits=%s ScalerCrop=%s",
                            fd,
                            fdl,
                            sc,
                        )
                    except Exception:
                        pass
                    logged_camera_controls = True
                cv2.putText(frame_display, "Capture FPS:{:.1f} [Hz]".format(camera.measured_fps),
                    textLocation0, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, "Display FPS:{:.1f} [Hz]".format(measured_dps),
                    textLocation1, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, f"Mode:{configs.get('mode')}",
                    textLocation2, font, fontScale, fontColor, lineType,
                )

                cv2.imshow(window_name, frame_display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                # if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                #    stop = True

                last_display = current_time
                num_frames_displayed += 1

            # Update display FPS measurement
            delta_dps = current_time - last_dps_time
            if delta_dps >= dps_measure_interval:
                measured_dps = num_frames_displayed / delta_dps
                num_frames_displayed = 0
                last_dps_time = current_time

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

""""
(env) uutzinger@urspi:~/pythonBME210/camera $ python3 examples/picamera2_capture_display.py 
INFO:picamera2.picamera2:Initialization successful.
INFO:picamera2.picamera2:Camera now open.
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera started
INFO:PiCamera2 Capture:Getting Images
INFO:PiCamera2 Capture:Config: mode=main format=BGR3 camera_res=(640, 480) output_res=(-1, -1)
INFO:PiCamera2 Capture:PiCam2:MAIN sensor selection policy=maximize_fps desired_main=640x480 selected_sensor=(640, 480) bit_depth=10 fps~58.92
INFO:PiCamera2 Capture:PiCam2:ISP configuration successful - hardware will handle format=BGR888, size=(640, 480), transform=False
INFO:PiCamera2 Capture:PiCam2:Controls set {'AeEnable': True, 'AeMeteringMode': 0, 'AwbEnable': True, 'AwbMode': 0}
INFO:PiCamera2 Capture:PiCam2:Open summary stream=main size=(640, 480) fmt=BGR888 req_fps=60 FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) cpu_resize=False cpu_flip=False cpu_convert=False
INFO:PiCamera2 Capture:PiCam2:Camera opened
INFO:PiCamera2 Capture:PiCam2:Main Stream mode 640x480 format=BGR888. Supported main formats: XBGR8888, XRGB8888, RGB888, BGR888, YUV420, YUYV, MJPEG
INFO:PiCamera2 Capture:PiCam2:Main Stream can scale to arbitrary resolutions; non-native aspect ratios may crop. For raw modes list, run examples/list_Picamera2Properties.py.
INFO:PiCamera2 Capture:PiCam2:Suggested Main Stream options (camera_res/output_res, max_fps, full_fov):
INFO:PiCamera2 Capture:PiCam2:  640x480 -> 640x480 fmt=BGR888 max_fps~58.9 full_fov=False
INFO:PiCamera2 Capture:PiCam2:  1296x972 -> 1296x972 fmt=BGR888 max_fps~46.3 full_fov=False
INFO:PiCamera2 Capture:PiCam2:  1920x1080 -> 1920x1080 fmt=BGR888 max_fps~32.8 full_fov=False
INFO:PiCamera2 Capture:PiCam2:  2592x1944 -> 2592x1944 fmt=BGR888 max_fps~15.6 full_fov=True
INFO:PiCamera2 Capture:PiCam2:=== camera configuration ===
INFO:PiCamera2 Capture:PiCam2:Requested mode=main camera_res=(640, 480) output_res=(-1, -1) format=BGR3 fps=60 stream_policy=maximize_fps low_latency=False flip=0
INFO:PiCamera2 Capture:PiCam2:Requested controls exposure=0 autoexposure=1 aemeteringmode=center autowb=1 awbmode=auto
INFO:PiCamera2 Capture:PiCam2:camera_configuration={'use_case': 'video', 'transform': <libcamera.Transform 'identity'>, 'colour_space': <libcamera.ColorSpace 'SMPTE170M'>, 'buffer_count': 6, 'queue': True, 'main': {'format': 'BGR888', 'size': (640, 480), 'preserve_ar': True, 'stride': 1920, 'framesize': 921600}, 'lores': None, 'raw': {'format': 'GBRG_PISP_COMP1', 'size': (640, 480), 'stride': 640, 'framesize': 307200}, 'controls': {'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}, 'sensor': {'bit_depth': 10, 'output_size': (640, 480)}, 'display': 'main', 'encode': 'main'}
INFO:PiCamera2 Capture:PiCam2:configured controls={'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}
INFO:PiCamera2 Capture:PiCam2:camera_properties={'Model': 'ov5647', 'UnitCellSize': (1400, 1400), 'Location': 2, 'Rotation': 0, 'ColorFilterArrangement': 2, 'PixelArraySize': (2592, 1944), 'PixelArrayActiveAreas': [(16, 6, 2592, 1944)], 'ScalerCropMaximum': (16, 0, 2560, 1920), 'SystemDevices': (20752, 20753, 20754, 20755, 20756, 20757, 20758, 20739, 20740, 20741, 20742), 'SensorSensitivity': 1.0}
INFO:PiCamera2 Capture:PiCam2:metadata FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) AeEnable=None ExposureTime=16836 AwbEnable=None AwbMode=None AeMeteringMode=None AnalogueGain=7.5
INFO:PiCamera2 Capture:Camera controls: FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920)
INFO:picamera2.picamera2:Camera stopped
INFO:picamera2.picamera2:Camera closed successfully.


=== camera configuration ===
Requested: mode=main size=(640, 480) format=BGR888 fps=60.0 stream_policy=default low_latency=False flip=0
Requested controls: {'FrameDurationLimits': (16667, 16667)}
camera_configuration: {'use_case': 'video', 'transform': <libcamera.Transform 'identity'>, 'colour_space': <libcamera.ColorSpace 'SMPTE170M'>, 'buffer_count': 6, 'queue': True, 'main': {'format': 'BGR888', 'size': (640, 480), 'preserve_ar': True, 'stride': 1920, 'framesize': 921600}, 'lores': None, 'raw': {'format': 'GBRG_PISP_COMP1', 'size': (640, 480), 'stride': 640, 'framesize': 307200}, 'controls': {'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}, 'sensor': {'bit_depth': 10, 'output_size': (640, 480)}, 'display': 'main', 'encode': 'main'}
camera_properties: {'Model': 'ov5647', 'UnitCellSize': (1400, 1400), 'Location': 2, 'Rotation': 0, 'ColorFilterArrangement': 2, 'PixelArraySize': (2592, 1944), 'PixelArrayActiveAreas': [(16, 6, 2592, 1944)], 'ScalerCropMaximum': (16, 0, 2560, 1920), 'SystemDevices': (20752, 20753, 20754, 20755, 20756, 20757, 20758, 20739, 20740, 20741, 20742), 'SensorSensitivity': 1.0}
metadata: FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) AeEnable=None ExposureTime=16836

"""