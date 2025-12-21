 
###############################################################################
# RTSP video capture
#
# Uses opencv video capture to capture rtsp stream
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
#
# 2025
# 2021 Initialize, remove frame access (use only queue)
# 2019 Initial release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

# System
import logging, time, platform

# Open Computer Vision
import cv2

###############################################################################
# RTSP Capture
###############################################################################

class rtspCapture(Thread):
    """
    This thread continually captures frames from an RTSP stream
    """

    # Initialize the Camera Thread
    # Opens Capture Device
    ############################################################################
    def __init__(self, configs, 
        rtsp: (str) = None, 
        gpu: (bool) = False,
        queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}
        
        # populate desired settings from configuration file or function call
        if rtsp is not None: 
            self._rtsp      = rtsp
        else: 
            self._rtsp      = self._configs.get('rtsp', None)
        self._output_res    = self._configs.get('output_res', (-1, -1))
        self._output_width  = self._output_res[0]
        self._output_height = self._output_res[1]
        self._flip_method   = self._configs.get('flip', 0)
        self._gpuavail      = gpu

        # Threading Locks, Events
        # Threading Queue, Locks, Events
        self.capture         = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True
        self.cam_lock        = Lock()

        # open up the camera
        self.open_cam()

        # Init Frame and Thread
        self.frame_time   = 0.0
        self.measured_fps = 0.0

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        # clean up

    def close_cam(self):
        """Release the underlying VideoCapture (idempotent)."""
        try:
            with self.cam_lock:
                cam = getattr(self, 'cam', None)
                if cam is not None:
                    cam.release()
                self.cam = None
            self.cam_open = False
        except Exception:
            pass

    def start(self, capture_queue = None):
        """start the capture thread"""
        self.stopped = False
        super().start()

    # Thread entrypoint
    def run(self):
        """thread entrypoint"""
        if not getattr(self, 'cam_open', False):
            return
        self.update()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_frames = 0
        try:
            while not self.stopped:
                current_time = time.time()

                if self.cam is None:
                    time.sleep(0.01)
                    continue

                with self.cam_lock:
                    ret, img = self.cam.read()
                if (not ret) or (img is None):
                    time.sleep(0.005)
                    continue

                num_frames += 1
                self.frame_time = int(current_time * 1000)

                if self.capture.full():
                    if not self.log.full():
                        self.log.put_nowait((logging.WARNING, "RTSPCap:Capture Queue is full!"))
                    continue

                img_proc = img

                # Resize only if an explicit output size was provided
                if (self._output_width > 0) and (self._output_height > 0):
                    img_proc = cv2.resize(img_proc, self._output_res)

                # Apply flip/rotation if requested (same enum as cv2Capture)
                if self._flip_method != 0:
                    if self._flip_method == 1: # ccw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 2: # rot 180
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_180)
                    elif self._flip_method == 3: # cw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_CLOCKWISE)
                    elif self._flip_method == 4: # horizontal (left-right)
                        img_proc = cv2.flip(img_proc, 1)
                    elif self._flip_method == 5: # upright diagonal. ccw & lr
                        img_proc = cv2.flip(cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                    elif self._flip_method == 6: # vertical (up-down)
                        img_proc = cv2.flip(img_proc, 0)
                    elif self._flip_method == 7: # upperleft diagonal
                        img_proc = cv2.transpose(img_proc)

                self.capture.put_nowait((self.frame_time, img_proc))

                # FPS calculation
                if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                    self.measured_fps = num_frames/5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "RTSPCAM:FPS:{}".format(self.measured_fps)))
                    num_frames = 0
                    last_time = current_time
        finally:
            self.close_cam()

    # Setup the Camera
    ############################################################################

    def open_cam(self):
        """
        Open up the camera so we can begin capturing 
        """

        if not self._rtsp:
            self.cam = None
            self.cam_open = False
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTSPCap:Missing 'rtsp' URL in configs and no rtsp argument provided"))
            return

        gst ='rtspsrc location=' + self._rtsp + ' latency=10 ! rtph264depay ! h264parse ! '
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                # Jetson decode + convert (ensure we end up in system memory)
                gst = gst + 'omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'avdec_h264 ! videoconvert ! '

        # Force a deterministic OpenCV-friendly format.
        # BGR is what cv2 expects; using drop/max-buffers avoids latency buildup.
        gst = gst + 'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'

        self.log.put_nowait((logging.INFO, gst))
        self.cam = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)

        self.cam_open = self.cam.isOpened()        
        if not self.cam_open:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "RTSPCap:Failed to open camera!"))

#
# Testing 
###################################################################
if __name__ == '__main__':

    configs = {
        'rtsp'            : 'rtsp://10.41.83.100:554/camera',
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 0,              # 0=norotation 
    }

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")

    logger.log(logging.DEBUG, "Starting Capture")

    camera = rtspCapture(configs)
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")

    window_handle = cv2.namedWindow('RTSP Camera', cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty('RTSP Camera', 0) >= 0):
        try:
            (frame_time, frame) = camera.capture.get()
            cv2.imshow('RTSP Camera', frame)
        except: pass

        if cv2.waitKey(1) & 0xFF == ord('q'):  break

        try: 
            (level, msg)=camera.log.get_nowait()
            logger.log(level, "RTSPCap:{}".format(msg))
        except: pass

    camera.stop()
    cv2.destroyAllWindows()
