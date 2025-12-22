###############################################################################
# RTSP video capture
#
# Cross-platform strategy:
# - Preferred backend (when available): GStreamer via PyGObject (gi) + appsink.
# - Default backend on Windows/macOS (pip-friendly): OpenCV VideoCapture on RTSP URL
#   (OpenCV PyPI wheels typically include FFmpeg).
# - Optional Linux fallback when OpenCV has GStreamer support: CAP_GSTREAMER pipeline.
#
# Urs Utzinger
#
# 2025: GI/appsink backend + cross-platform fallbacks
# 2021: Initialize, remove frame access (use only queue)
# 2019: Initial release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

# System
import logging, time, platform

# Array
import numpy as np

# Open Computer Vision
import cv2

# GStreamer (direct)
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstApp, GstVideo
except Exception:  # pragma: no cover
    Gst = None
    GstApp = None
    GstVideo = None

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

        # Backend preference
        # - prefer_gi: try GI/appsink first when available
        # Note: we intentionally do NOT auto-failover between backends at runtime.
        self._prefer_gi = bool(self._configs.get('prefer_gi', True))

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

        # Active backend identifier: 'gi' or 'opencv'
        self._backend = getattr(self, '_backend', None)

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        # clean up

    def close_cam(self):
        """Release the underlying capture backend (idempotent)."""
        # OpenCV capture
        try:
            with self.cam_lock:
                cam = getattr(self, 'cam', None)
                if cam is not None:
                    cam.release()
                self.cam = None
        except Exception:
            pass

        # GI pipeline
        if getattr(self, 'pipeline', None) is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None

        self.cam_open = False

    def _close_gi(self):
        if getattr(self, 'pipeline', None) is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None

    def _close_opencv(self):
        try:
            with self.cam_lock:
                cam = getattr(self, 'cam', None)
                if cam is not None:
                    cam.release()
                self.cam = None
        except Exception:
            self.cam = None

    def _open_opencv_rtsp(self) -> bool:
        """Open RTSP using OpenCV (typically FFmpeg backend on Windows/macOS)."""
        self._close_opencv()
        try:
            api = getattr(cv2, "CAP_FFMPEG", 0)
            if api:
                self.cam = cv2.VideoCapture(self._rtsp, apiPreference=api)
            else:
                self.cam = cv2.VideoCapture(self._rtsp)
            return bool(self.cam is not None and self.cam.isOpened())
        except Exception:
            self.cam = None
            return False

    def _open_gi_rtsp_pipeline(self) -> bool:
        """Open RTSP using GI GStreamer pipeline + appsink (H264)."""
        if Gst is None:
            return False

        self._close_gi()

        if not hasattr(self.__class__, "_gst_inited"):
            Gst.init(None)
            self.__class__._gst_inited = True

        gst = 'rtspsrc location=' + self._rtsp + ' latency=10 ! '
        gst = gst + 'rtph264depay ! h264parse ! '

        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64':
                gst = gst + 'omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
            elif platform.machine() in ('armv6l', 'armv7l'):
                gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'avdec_h264 ! videoconvert ! '

        gst = (
            gst
            + 'videoconvert ! video/x-raw,format=BGR '
            + '! queue leaky=downstream max-size-buffers=1 '
            + '! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true enable-last-sample=false qos=false'
        )

        if not self.log.full():
            self.log.put_nowait((logging.INFO, gst))

        try:
            self.pipeline = Gst.parse_launch(gst)
            self.appsink = self.pipeline.get_by_name("appsink")
            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            return ret != Gst.StateChangeReturn.FAILURE
        except Exception as exc:
            self.pipeline = None
            self.appsink = None
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, f"RTSPCap:GI pipeline open failed: {exc}"))
            return False

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

                img = None

                # Active backend read
                if self._backend == 'gi' and getattr(self, 'appsink', None) is not None:
                    try_pull = getattr(self.appsink, "try_pull_sample", None)
                    if callable(try_pull):
                        sample = try_pull(int(250 * 1e6))  # 250ms
                    else:
                        sample = self.appsink.emit("try-pull-sample", int(250 * 1e6))
                    if sample is not None:
                        try:
                            img = self._sample_to_bgr(sample)
                        except Exception as exc:
                            img = None
                            if not self.log.full():
                                self.log.put_nowait((logging.ERROR, f"RTSPCap:Failed to convert sample to BGR: {exc}"))

                if self._backend == 'opencv' and getattr(self, 'cam', None) is not None:
                    with self.cam_lock:
                        ret, img_cv = self.cam.read()
                    if ret and (img_cv is not None):
                        img = img_cv

                if img is None:
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

        self.pipeline = None
        self.appsink  = None
        self.cam      = None
        self.cam_open = False

        # Choose initial backend (do not open both at once)
        self._backend = None
        opened = False
        if self._prefer_gi and Gst is not None:
            opened = self._open_gi_rtsp_pipeline()
            if opened:
                self._backend = 'gi'
                self.cam_open = True

        if not opened:
            opened = self._open_opencv_rtsp()
            if opened:
                self._backend = 'opencv'
                self.cam_open = True

        # Linux extra fallback: OpenCV + CAP_GSTREAMER pipeline (useful on Jetson builds)
        if not opened and platform.system() == "Linux":
            gst = 'rtspsrc location=' + self._rtsp + ' latency=10 ! rtph264depay ! h264parse ! '
            if platform.machine() == 'aarch64':
                gst = gst + 'omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
            elif platform.machine() in ('armv6l', 'armv7l'):
                gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
            else:
                gst = gst + 'avdec_h264 ! videoconvert ! '
            gst = gst + 'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
            if not self.log.full():
                self.log.put_nowait((logging.INFO, gst))
            try:
                self.cam = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)
                opened = self.cam.isOpened()
                if opened:
                    self._backend = 'opencv'
                    self.cam_open = True
            except Exception:
                self.cam = None
                opened = False

        if not self.cam_open:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTSPCap:Failed to open camera!"))

    def _sample_to_bgr(self, sample):
        """Convert a Gst.Sample (video/x-raw,format=BGR) into a NumPy array."""
        if sample is None or GstVideo is None:
            return None

        caps = sample.get_caps()
        if caps is None:
            return None

        info_new = getattr(GstVideo.VideoInfo, "new_from_caps", None)
        if callable(info_new):
            info = info_new(caps)
            if info is None:
                return None
        else:
            info = GstVideo.VideoInfo()
            ok = info.from_caps(caps)
            if not ok:
                return None

        buf = sample.get_buffer()
        if buf is None:
            return None

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            height = int(info.height)
            width = int(info.width)
            stride = int(info.stride[0]) if info.stride and info.stride[0] else width * 3

            data = np.frombuffer(map_info.data, dtype=np.uint8)
            if data.size < height * stride:
                return None

            frame_2d = data[: height * stride].reshape((height, stride))
            frame = frame_2d[:, : width * 3].reshape((height, width, 3))
            return frame.copy()
        finally:
            buf.unmap(map_info)

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
