##########################################################################
# Testing of display and capture using PiCamera2 on Raspberry Pi
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
# Functions and Classes
##########################################################################

##########################################################################
# Initialize
##########################################################################

# Setting up logging ----

logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Raspi Capture")

# Silence Picamera2 / libcamera logs; keep only this script's logging output
for _name in ("picamera2", "libcamera"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# Configs and Variables ----

# default camera starts at 0 by operating system
camera_index = 0

configs = {
    ##############################################
    # Picamera2 capture configuration
    #
    # List the camera properties with:
    #     examples/list_Picamera2Properties.py
    #
    # Core capture mode:
    #   'main' -> full-FOV processed stream (BGR/YUV), scaled to 'camera_res' (libcamera scales)
    #   'raw'  -> high-FPS raw sensor window (exact sensor mode only), cropped FOV
    ##############################################
    'mode'            : 'main',         # 'main' or 'raw'
    # Resolution and frame rate
    'camera_res'      : (640, 480),     # requested main stream size (w, h)
    # 'raw_res'       : (640, 480),     # optional raw sensor window (w, h); defaults to camera_res
    'output_res'      : (-1, -1),       # (-1, -1): output == input; else libcamera scales main to this
    'fps'             : 120,            # requested frame rate
    # Exposure / Auto-exposure
    'exposure'        : 0,              # manual ExposureTime in microseconds; 0/-1 -> leave AE in charge
    'autoexposure'    : 1,              # -1: leave unchanged, 0: AE off, 1: AE on
    # AE metering: int or friendly string ('center', 'spot', 'matrix')
    'aemeteringmode' : 'center',        # default: CentreWeighted (0) if omitted
    # Auto White Balance
    'autowb'          : 1,              # -1: leave unchanged, 0: AWB off, 1: AWB on
    # AWB mode: int or friendly string ('auto', 'tungsten', 'fluorescent', 'indoor', 'daylight', 'cloudy')
    'awbmode'        : 'auto',          # default: Auto (0) if omitted
    # Formats
    # Main Stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
    # Raw Streanm formats: SRGGB8, SRGGB10_CSI2P (see properties script)
    'format'          : 'BGR3',         # legacy combined format
    # 'main_format'   : 'BGR3',         # optional explicit main format override
    # 'raw_format'    : 'SRGGB8',       # optional explicit RAW format override
    # Sensor-mode selection policy
    #   'default' / 'maximize_fov' : prefer widest FOV (for main)
    #   'maximize_fps'             : prefer highest FPS
    'stream_policy'  : 'default',
    # Low-latency options
    #   low_latency=True  -> small camera buffer_count and size-1 queue (latest frame)
    #   low_latency=False -> Picamera2 default buffer_count and normal queue
    'low_latency'     : False,
    # Optional explicit overrides (advanced):
    # 'buffer_count'  : 3,              # libcamera buffer_count passed to Picamera2
    # Flip and transforms
    'flip'            : 0,              # 0..7 as in README Video Flip table
    # 'hw_transform'  : 1,              # 1: try libcamera transform, 0: always use CPU flip
    # Queues and display
    # 'buffersize'    : 4,              # capture queue size; if omitted and low_latency=True -> 1
    'displayfps'      : 10              # frame rate for display server (UI update)
}

# Display ----

if configs['displayfps'] >= 0.8*configs['fps']:
    display_interval = 0
else:
    display_interval = 1.0/configs['displayfps']

dps_measure_time = 5.0 # assess performance every 5 secs

window_name      = 'Camera'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,40)
textLocation2    = (10,100)
fontScale        = 0.5
fontColor        = (255,255,255)
lineType         = 1

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Camera ----

# Create camera interface based on computer OS you are running
# Prefer Raspberry Pi Picamera2/libcamera when available, otherwise fall back to OpenCV.
camera = None
try:
    from camera.capture.picamera2capture import piCamera2Capture

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not getattr(camera, 'cam_open', False):
        camera = None
except Exception:
    camera = None

if camera is None:
    from camera.capture.cv2capture import cv2Capture

    camera = cv2Capture(configs, camera_index)

logger.log(logging.INFO, "Getting Images")
logger.log(
    logging.INFO,
    "Config: mode=%s format=%s camera_res=%s output_res=%s",
    configs.get('mode'), configs.get('format'), configs.get('camera_res'), configs.get('output_res')
)

camera.start()

# Initialize Loop
last_display   = time.perf_counter()
last_fps_time  = time.perf_counter()
measured_dps   = 0
num_frames_received    = 0
num_frames_displayed   = 0

stop = False
try:
    while(not stop):

        current_time = time.perf_counter()

        # wait for new image (timeout keeps UI responsive even if capture stalls)
        try:
            (frame_time, frame) = camera.capture.get(timeout=0.25)
            num_frames_received += 1
            # Convert using picamera2capture helper directly to OpenCV BGR
            frame = camera.convert(frame, to='BGR888')
        except Empty:
            frame = None

        # display log
        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "{}".format(msg))

        # calc stats
        if (current_time - last_fps_time) >= dps_measure_time:
            measured_fps = num_frames_received/dps_measure_time
            logger.log(logging.INFO, "MAIN:Frames received per second:{}".format(measured_fps))
            num_frames_received = 0
            measured_dps = num_frames_displayed/dps_measure_time
            logger.log(logging.INFO, "MAIN:Frames displayed per second:{}".format(measured_dps))
            num_frames_displayed = 0
            last_fps_time = current_time

        # analyze your frames here
        # .....

        # display (at slower rate than capture)
        if (frame is not None) and ((current_time - last_display) >= display_interval):
            # If the window was closed, stop before calling imshow (prevents recreation)
            try:
                window_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
            except Exception:
                window_visible = False
            if not window_visible:
                stop = True
            else:
                frame_display = frame.copy()
                cv2.putText(frame_display, "Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation0, font, fontScale, fontColor, lineType)
                cv2.putText(frame_display, "Display FPS:{} [Hz]".format(measured_dps),        textLocation1, font, fontScale, fontColor, lineType)
                try:
                    cv2.putText(frame_display, f"Mode:{configs.get('mode')}",                 textLocation2, font, fontScale, fontColor, lineType)
                except Exception:
                    pass
                cv2.imshow(window_name, frame_display)

                # quit the program if users enter q or closes the display window
                # the waitKey function limits the display frame rate
                # without waitKey the opencv window is not refreshed
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
