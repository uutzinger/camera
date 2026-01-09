##########################################################################
# RAW capture with PiCamera2 on Raspberry Pi
#
# - Uses piCamera2Capture in RAW Bayer mode
# - stream_policy='maximize_fps' to prefer highest-FPS sensor mode
# - low_latency=True for minimal buffering (small buffer_count, size-1 queue)
#
# Actual RAW resolution and FPS are determined by the camera's sensor modes.
# The selected RAW mode, all available RAW sensor modes, and any FOV cropping
# are reported via the piCamera2Capture.log queue.
##########################################################################

import cv2
import logging
import time

from queue import Empty

# Optimize OpenCV performance on small CPUs
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)
except Exception:
    pass

##########################################################################
# Initialize
##########################################################################

logging.basicConfig(level=logging.INFO)  # options: DEBUG, INFO, WARNING, ERROR
logger = logging.getLogger("Raspi RAW MaxFPS")

# default camera starts at 0 by operating system
camera_index = 0

# -----------------------------------------------------------------------
# PiCamera2 RAW + Max-FPS + Low-Latency configuration
# -----------------------------------------------------------------------

configs = {
    ##############################################
    # PiCamera2 RAW Bayer capture configuration
    #
    # List the camera properties and sensor modes with:
    #   examples/list_Picamera2Properties.py
    ##############################################
\    'mode'            : 'raw',
    'camera_res'      : (640, 480),     # hint resolution (w, h)
    'output_res'      : (-1, -1),       # RAW: keep sensor mode size; avoid extra CPU resize
    'fps'             : 120,            # requested frame rate (sensor modes / exposure may limit this)
    'exposure'        : 0,              # manual ExposureTime [us]; 0/-1 -> AE in charge
    'autoexposure'    : 1,              # -1: leave, 0: AE off, 1: AE on
    'aemeteringmode' : 'center',
    'autowb'          : 1,              # -1: leave, 0: AWB off, 1: AWB on
    'awbmode'        : 'auto',
    'raw_format'      : 'SRGGB8',
    'stream_policy'  : 'maximize_fps',
    'low_latency'     : True,
    'flip'            : 0,              # no rotation/flip for maximum throughput
    'displayfps'      : 15,             # UI update rate; capture FPS may be higher
}

# Display timing ----

if configs['displayfps'] >= 0.8 * configs['fps']:
    display_interval = 0.0
else:
    display_interval = 1.0 / configs['displayfps']

dps_measure_time = 5.0  # assess performance every 5 secs

window_name      = 'PiCamera2 RAW MaxFPS'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10, 20)
textLocation1    = (10, 40)
textLocation2    = (10, 60)
fontScale        = 0.5
fontColor        = (255, 255, 255)
lineType         = 1

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Camera (PiCamera2-only) ----

camera = None
try:
    from camera.capture.picamera2capture import piCamera2Capture

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not getattr(camera, 'cam_open', False):
        logger.error("PiCam2:Failed to open camera in RAW mode; see logs above.")
        camera = None
except Exception as exc:
    logger.error("PiCam2:piCamera2Capture import or init failed: %s", exc)
    camera = None

if camera is None:
    logger.error("PiCam2:RAW example requires PiCamera2; cv2 fallback is not supported here.")
    raise SystemExit(1)

logger.info("Getting RAW images (mode=raw, stream_policy=maximize_fps, low_latency=True)")
logger.info(
    "Config: mode=%s raw_format=%s camera_res=%s output_res=%s fps=%s",
    configs.get('mode'), configs.get('raw_format'), configs.get('camera_res'), configs.get('output_res'), configs.get('fps')
)

camera.start()

# Initialize Loop
last_display   = time.perf_counter()
last_fps_time  = time.perf_counter()
measured_dps   = 0.0
num_frames_received    = 0
num_frames_displayed   = 0

stop = False
try:
    while not stop:
        current_time = time.perf_counter()

        # wait for new image (timeout keeps UI responsive even if capture stalls)
        try:
            (frame_time, frame) = camera.capture.get(timeout=0.25)
            num_frames_received += 1
            # Convert RAW Bayer to OpenCV BGR for display
            frame = camera.convert(frame, to='BGR888')
        except Empty:
            frame = None

        # display log
        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "%s", msg)

        # calc stats
        if (current_time - last_fps_time) >= dps_measure_time:
            measured_fps = num_frames_received / dps_measure_time
            logger.info("RAW:Frames received per second: %s", measured_fps)
            num_frames_received = 0
            measured_dps = num_frames_displayed / dps_measure_time
            logger.info("RAW:Frames displayed per second: %s", measured_dps)
            num_frames_displayed = 0
            last_fps_time = current_time

        # display (at slower rate than capture)
        if (frame is not None) and ((current_time - last_display) >= display_interval):
            try:
                window_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
            except Exception:
                window_visible = False
            if not window_visible:
                stop = True
            else:
                frame_display = frame.copy()
                cv2.putText(frame_display, f"Capture FPS:{camera.measured_fps:.1f} [Hz]", textLocation0, font, fontScale, fontColor, lineType)
                cv2.putText(frame_display, f"Display FPS:{measured_dps:.1f} [Hz]",      textLocation1, font, fontScale, fontColor, lineType)
                mode_info = f"Mode:raw Policy:maximize_fps Low-latency:True"
                cv2.putText(frame_display, mode_info,                                  textLocation2, font, fontScale, fontColor, lineType)
                cv2.imshow(window_name, frame_display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop = True
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        stop = True
                except Exception:
                    stop = True
                last_display = current_time
                num_frames_displayed += 1

finally:
    # Clean up
    try:
        camera.stop()
        camera.join(timeout=2.0)
        camera.close_cam()
    except Exception:
        pass

    cv2.destroyAllWindows()
