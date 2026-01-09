##########################################################################
# Testing of display, analysis and capture using PiCamera2 on Raspberry Pi
##########################################################################
import os
import cv2
import logging
import time
import numpy as np
from typing import Any

# Prefer tflite_runtime, fall back to tensorflow.lite
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter  # type: ignore
    except Exception:
        Interpreter = None  # type: ignore
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

def load_interpreter(model_path: str) -> Any:
    """
    Loads a TensorFlow Lite convolutional neural network (CNN) model
    into memory and prepares it for inference.

    Parameters
    ----------
    model_path : str
        File path to the .tflite MoveNet model.

    Returns
    -------
    Interpreter
        A TensorFlow Lite interpreter ready to run inference.
    """
    if Interpreter is None:
        raise ImportError("No TFLite Interpreter found. Install tflite-runtime or tensorflow.")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_bgr(frame_bgr: np.ndarray, input_h: int, input_w: int, input_dtype) -> np.ndarray:
    """
    Preprocesses a camera frame so it matches the input requirements
    of the MoveNet neural network.

    This includes:
    - Converting from BGR (OpenCV default) to RGB
    - Resizing to the model's expected resolution
    - Casting to the correct data type
    - Adding a batch dimension

    Parameters
    ----------
    frame_bgr : np.ndarray
        Input image frame from OpenCV in BGR format.
    input_h : int
        Height expected by the neural network.
    input_w : int
        Width expected by the neural network.
    input_dtype :
        Data type required by the model input tensor.

    Returns
    -------
    np.ndarray
        Preprocessed image tensor ready for neural network inference.
    """
    # MoveNet examples use RGB input; many OpenCV cameras provide BGR
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_w, input_h), interpolation=cv2.INTER_AREA)

    # Most MoveNet int8 models expect uint8; but we handle dtype generically
    if input_dtype == np.uint8:
        inp = resized.astype(np.uint8)
    else:
        # float models typically expect float32
        inp = resized.astype(np.float32)
        # Some float models expect [0,1] normalization; check your model if needed
        inp = inp / 255.0

    return np.expand_dims(inp, axis=0)  # [1, H, W, 3]

def infer_keypoints(interpreter: Any, input_tensor: np.ndarray) -> np.ndarray:
    """
    Runs the MoveNet neural network on a preprocessed input image
    and extracts the predicted human pose keypoints.

    Parameters
    ----------
    interpreter : Interpreter
        Loaded TensorFlow Lite interpreter.
    input_tensor : np.ndarray
        Preprocessed input image tensor.

    Returns
    -------
    np.ndarray
        Raw model output containing normalized keypoint coordinates
        and confidence scores.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    # [1, 1, 17, 3]
    return interpreter.get_tensor(output_details[0]["index"])

def keypoints_to_pixels(kpts: np.ndarray, frame_h: int, frame_w: int):
    """
    Converts normalized keypoint coordinates (0–1) produced by MoveNet
    into pixel coordinates relative to the original camera frame.

    Parameters
    ----------
    kpts : np.ndarray
        Raw keypoint output from the neural network.
    frame_h : int
        Height of the camera frame in pixels.
    frame_w : int
        Width of the camera frame in pixels.

    Returns
    -------
    list of tuples
        List of (x, y, confidence) keypoints in pixel coordinates.
    """
    # kpts shape: [1, 1, 17, 3] -> [17, 3]
    pts = kpts[0, 0, :, :]  # (y, x, score)
    out = []
    for (y, x, s) in pts:
        px = int(x * frame_w)
        py = int(y * frame_h)
        out.append((px, py, float(s)))
    return out

def draw_pose(frame_bgr: np.ndarray, points, min_score=0.2):
    """
    Draws detected human pose keypoints and skeletal connections
    onto the camera frame using OpenCV drawing primitives.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Original camera frame.
    points : list of tuples
        List of (x, y, confidence) keypoints in pixel coordinates.
    min_score : float
        Minimum confidence threshold required to draw a keypoint
        or skeletal connection.
    """
    # points: list of (x,y,score)
    for (x, y, s) in points:
        if s >= min_score:
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)

    for (a, b) in EDGES:
        xa, ya, sa = points[a]
        xb, yb, sb = points[b]
        if sa >= min_score and sb >= min_score:
            cv2.line(frame_bgr, (xa, ya), (xb, yb), (255, 0, 0), 2)

##########################################################################
# Initialize
##########################################################################

# Setting up logging -----

logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Raspi Capture")

# Silence Picamera2 / libcamera logs; keep only this script's logging output
for _name in ("picamera2", "libcamera"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
# Also silence libcamera C++ logs via environment (must be set before libcamera loads)
os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:0")  # 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG
    
# Configs and Variables ----

# You can obtain the MoveNet model from TensorFlow Hub:
# fast model
# wget -O model.tflite "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
# or more accurate model
# wget -O model.tflite "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite"

model_path = "model.tflite"

# default camera starts at 0 by operating system
camera_index = 0

configs = {
    ##############################################
    # list the camera properties with 
    #     list_Picamera2Properties.py
    ##############################################
    # Capture mode:
    #   'main' -> full-FOV processed stream (BGR/YUV), scaled to 'camera_res' (libcamera scales)
    #   'raw'  -> high-FPS raw sensor window (exact sensor mode only), cropped FOV
    'mode'           : 'preview',
    'camera_res'      : (640, 480),     # unified resolution
    'exposure'        : 0,              # microseconds, 0/-1 for auto
    'autoexposure'    : 1,              # 0=manual, 1=auto
    'autowb'          : 1,              # 0=disable, 1=enable
    'fps'             : 120,            # 
    # Preview formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
    # Raw formats:     SRGGB8, SRGGB10_CSI2P, (see properties script)
    'format'         : 'BGR3',
    'buffersize'      : 4,              # default is 4 for V4L2, max 10, 
    'output_res'      : (-1, -1),       # output resolution same as input 
    'flip'            : 0,              # 0=norotation 
    'displayfps'      : 10              # frame rate for display server
}

# COCO-17 keypoints used by MoveNet SinglePose
# 
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton edges (index pairs) — matches common MoveNet demo conventions
#
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

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

# Load the model ----

if not os.path.exists(model_path):
    logger.log(logging.CRITICAL, f"Model file not found: {model_path}")
    raise SystemExit(1)
interpreter = load_interpreter(model_path)

# Extract model details
input_details = interpreter.get_input_details()
_, in_h, in_w, _ = input_details[0]["shape"]
in_dtype = input_details[0]["dtype"]

# Camera -----

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

# Display -----

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Initialize Loop -----

last_display   = time.perf_counter()
last_fps_time  = time.perf_counter()
measured_dps   = 0
num_frames_received    = 0
num_frames_displayed   = 0
stop = False

# Loop -----

try:
    while(not stop):

        current_time = time.perf_counter()

        # Wait for new image (timeout keeps UI responsive even if capture stalls)
        try:
            (frame_time, frame) = camera.capture.get(timeout=0.25)
            num_frames_received += 1
            # Convert using picamera2capture helper directly to OpenCV BGR
            frame = camera.convert(frame, to='BGR888')
        except Empty:
            frame = None

        # Display log
        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "{}".format(msg))

        # Calc stats
        if (current_time - last_fps_time) >= dps_measure_time:
            measured_fps = num_frames_received/dps_measure_time
            logger.log(logging.INFO, "MAIN:Frames received per second:{}".format(measured_fps))
            num_frames_received = 0
            measured_dps = num_frames_displayed/dps_measure_time
            logger.log(logging.INFO, "MAIN:Frames displayed per second:{}".format(measured_dps))
            num_frames_displayed = 0
            last_fps_time = current_time

        # Analysis
        if frame is not None:
            # Convert frame to model input
            h, w = frame.shape[:2]
            input_tensor = preprocess_bgr(frame, in_h, in_w, in_dtype)
            # Infer the keypoints
            kpts = infer_keypoints(interpreter, input_tensor)
            points = keypoints_to_pixels(kpts, h, w)
            # Draw the keypoints
            draw_pose(frame, points, min_score=0.2)

        # Display (at slower rate than capture)
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
                # waitKey also needed to update imshow window
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
