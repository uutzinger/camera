###############################################################################
# RTP point to point video capture
#
# Captures an RTP (UDP) stream.
# - Preferred backend: GStreamer via PyGObject (gi) + appsink (like gCapture)
# - Fallback backend: 
#     OpenCV VideoCapture (CAP_GSTREAMER) if GI is unavailable
#     FFmpeg subprocess (via imageio-ffmpeg) if GI and OpenCV option are unavailable
#
# Urs Utzinger
#
# Changes
# 2025: GI/appsink backend added, FFMPEG fallback added
# 2021: Initial release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from queue import Queue
from threading import Thread, Lock

# System
import logging, time, platform

# Subprocess (FFmpeg fallback)
import subprocess

# Array
import numpy as np

# Open Computer Vision (used for resize/flip and as fallback capture backend)
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

# FFmpeg helper (bundled binary via pip)
try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None

###############################################################################
# RTP Capture
###############################################################################

class rtpCapture(Thread):
    """
    This thread continually captures frames from a point to point RTP stream
    This is not capturing rtsp.
    """

    # Initialize the Camera Thread
    # Opens Capture Device
    ############################################################################
    def __init__(
        self,
        configs=None,
        port: int = None,
        gpu: bool = False,
        queue_size: int = 32,
    ):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}

        # populate settings
        #########################################################
        if port is not None:
            self._port = int(port)
        else:
            self._port = int(self._configs.get('port', 554))
        self._gpuavail       = gpu
        self._output_res     = self._configs.get('output_res', (-1, -1))
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._flip_method    = self._configs.get('flip', 0)

        # Backend preference
        # - prefer_gi: try GI/appsink first when available
        # Note: we intentionally do NOT auto-failover between backends at runtime.
        self._prefer_gi = bool(self._configs.get('prefer_gi', True))

        # RTP caps defaults (keep legacy behavior)
        self._rtp_encoding_name = self._configs.get('rtp_encoding_name', 'H264')
        self._rtp_payload = int(self._configs.get('rtp_payload', 96))
        self._rtp_clock_rate = int(self._configs.get('rtp_clock_rate', 90000))

        # Threading Locks, Events
        self.log             = Queue(maxsize=32)
        self.capture         = Queue(maxsize=queue_size)
        self.stopped         = True
        self.cam_lock        = Lock()

        # open up the stream
        self.open_cam()

        # Init vars
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        # Active backend identifier: 'gi' | 'opencv' | 'ffmpeg'
        self._backend = getattr(self, '_backend', None)


    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def close_cam(self):
        """Release the underlying capture backend (idempotent)."""
        # OpenCV fallback
        try:
            cam = getattr(self, 'cam', None)
            if cam is not None:
                cam.release()
        except Exception:
            pass
        self.cam = None

        # FFmpeg subprocess
        proc = getattr(self, '_ffmpeg_proc', None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        self._ffmpeg_proc = None

        # GI pipeline
        if getattr(self, 'pipeline', None) is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None

        self.cam_open = False

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
                                self.log.put_nowait((logging.ERROR, f"RTPCap:Failed to convert sample to BGR: {exc}"))

                if self._backend == 'opencv' and getattr(self, 'cam', None) is not None:
                    with self.cam_lock:
                        ret, img_cv = self.cam.read()
                    if ret and (img_cv is not None):
                        img = img_cv

                if self._backend == 'ffmpeg':
                    img = self._ffmpeg_read_frame()

                if img is None:
                    time.sleep(0.005)
                    continue

                num_frames += 1
                self.frame_time = int(current_time * 1000)

                if not self.capture.full():
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
                else:
                    if not self.log.full():
                        self.log.put_nowait((logging.WARNING, "RTPCap:Capture Queue is full!"))

                # FPS calculation
                if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                    self.measured_fps = num_frames / 5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "RTPCAP:FPS:{}".format(self.measured_fps)))
                    last_time = current_time
                    num_frames = 0
        finally:
            self.close_cam()

    # Setup the Camera
    ############################################################################
    def open_cam(self):
        """
        Open up the camera so we can begin capturing frames
        """

        ## Open the camera with platform optimal settings
        # https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/

        gst = (
            'udpsrc port={:d} caps=application/x-rtp,media=(string)video,clock-rate=(int){:d},'
            'encoding-name=(string){:s},payload=(int){:d} ! '
        ).format(self._port, self._rtp_clock_rate, self._rtp_encoding_name, self._rtp_payload)

        # Depayload based on encoding (legacy only supported H264)
        if str(self._rtp_encoding_name).upper() == 'H264':
            gst = gst + 'rtph264depay ! '
        else:
            # Keep behavior explicit: unsupported encodings must be provided via custom pipeline
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"RTPCap:Unsupported rtp_encoding_name '{self._rtp_encoding_name}'. Only H264 is built-in."))
            self.cam_open = False
            return
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                # Jetson decode + convert (ensure we end up in system memory)
                gst = gst + 'h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = gst + 'h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'decodebin ! videoconvert ! '

        # Force a deterministic format (BGR) at the appsink boundary.
        gst = (
            gst
            + 'videoconvert ! video/x-raw,format=BGR '
            + '! queue leaky=downstream max-size-buffers=1 '
            + '! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true enable-last-sample=false qos=false'
        )
        if not self.log.full():
            self.log.put_nowait((logging.INFO, gst))

        # Prefer GI-based pipeline (matches gCapture behavior)
        self.pipeline = None
        self.appsink = None
        self.cam = None
        self.cam_open = False
        self._backend = None
        self._ffmpeg_proc = None

        if self._prefer_gi and Gst is not None:
            if not hasattr(self.__class__, "_gst_inited"):
                Gst.init(None)
                self.__class__._gst_inited = True
            try:
                self.pipeline = Gst.parse_launch(gst)
                self.appsink = self.pipeline.get_by_name("appsink")
                if self.appsink is None:
                    raise RuntimeError("appsink element not found in pipeline")

                # Ensure low-latency behavior even if pipeline is edited
                try:
                    self.appsink.set_property("sync", False)
                    self.appsink.set_property("drop", True)
                    self.appsink.set_property("max-buffers", 1)
                    if self.appsink.find_property("enable-last-sample") is not None:
                        self.appsink.set_property("enable-last-sample", False)
                    if self.appsink.find_property("qos") is not None:
                        self.appsink.set_property("qos", False)
                except Exception:
                    pass

                ret = self.pipeline.set_state(Gst.State.PLAYING)
                self.cam_open = ret != Gst.StateChangeReturn.FAILURE
                if self.cam_open:
                    self._backend = 'gi'
            except Exception as exc:
                self.pipeline = None
                self.appsink = None
                self.cam_open = False
                if not self.log.full():
                    self.log.put_nowait((logging.CRITICAL, f"RTPCap:Failed to open GI GStreamer pipeline: {exc}"))

        # Fallback 1: OpenCV VideoCapture (CAP_GSTREAMER)
        # NOTE: this requires OpenCV to be built with GStreamer, which is often NOT
        # true for pip wheels on Windows/macOS.
        if not self.cam_open:
            try:
                self.cam = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                self.cam_open = self.cam.isOpened()
                if self.cam_open:
                    self._backend = 'opencv'
            except Exception:
                self.cam = None
                self.cam_open = False

        # Fallback 2: FFmpeg subprocess (pip-friendly on Windows/macOS via imageio-ffmpeg)
        # Requires an SDP file so FFmpeg knows the RTP payload/clock/codec.
        if not self.cam_open:
            opened = self._open_ffmpeg_rtp()
            if opened:
                self.cam_open = True
                self._backend = 'ffmpeg'

        if not self.cam_open:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTPCap:Failed to open rtp stream!"))

    def _open_ffmpeg_rtp(self) -> bool:
        """Open RTP via ffmpeg subprocess.

        This is intended as a cross-platform fallback (esp. Windows/macOS) when
        GI/GStreamer isn't available.

        Required config:
          - 'rtp_sdp' (or 'sdp'): path to an SDP file describing the RTP stream.
          - 'camera_res' or 'output_res': used to size rawvideo reads.

                Example (FFmpeg fallback):
                        configs = {
                                'rtp_sdp': 'examples/rtp_h264_pt96.sdp',
                                'camera_res': (640, 480),
                        }
                        cap = rtpCapture(configs=configs, port=554)
        """

        if imageio_ffmpeg is None:
            return False

        sdp_path = self._configs.get('rtp_sdp', self._configs.get('sdp', None))
        if not sdp_path:
            # Without SDP, FFmpeg cannot reliably decode raw RTP because payload details are missing.
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTPCap:FFmpeg fallback requires configs['rtp_sdp'] (SDP file path)."))
            return False

        # Determine expected frame size for rawvideo reads
        out_res = self._configs.get('output_res', (-1, -1))
        cam_res = self._configs.get('camera_res', (-1, -1))
        if out_res[0] > 0 and out_res[1] > 0:
            width, height = int(out_res[0]), int(out_res[1])
        elif cam_res[0] > 0 and cam_res[1] > 0:
            width, height = int(cam_res[0]), int(cam_res[1])
        else:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTPCap:FFmpeg fallback needs configs['camera_res'] or configs['output_res'] to read raw frames."))
            return False

        self._ffmpeg_width = width
        self._ffmpeg_height = height
        self._ffmpeg_frame_bytes = width * height * 3

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Low-latency-ish defaults; produce raw BGR frames to stdout.
        cmd = [
            ffmpeg_exe,
            '-loglevel', 'error',
            '-hide_banner',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-analyzeduration', '0',
            '-probesize', '32',
            '-protocol_whitelist', 'file,udp,rtp',
            '-i', str(sdp_path),
            '-an',
            '-vf', f'scale={width}:{height}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            'pipe:1',
        ]

        if not self.log.full():
            self.log.put_nowait((logging.INFO, f"RTPCap:FFmpeg cmd: {' '.join(cmd)}"))

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._start_ffmpeg_stderr_reader(self._ffmpeg_proc, prefix="RTPCap:FFmpeg")
            return self._ffmpeg_proc.stdout is not None
        except Exception as exc:
            self._ffmpeg_proc = None
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"RTPCap:Failed to start FFmpeg: {exc}"))
            return False

    def _start_ffmpeg_stderr_reader(self, proc, prefix: str):
        """Drain FFmpeg stderr in a background thread to prevent pipe blockage."""

        try:
            stderr = getattr(proc, 'stderr', None)
            if stderr is None:
                return

            def _reader():
                try:
                    for raw in iter(stderr.readline, b''):
                        if not raw:
                            break
                        try:
                            line = raw.decode(errors='replace').strip()
                        except Exception:
                            line = str(raw)
                        if not line:
                            continue
                        if not self.log.full():
                            self.log.put_nowait((logging.ERROR, f"{prefix}:{line}"))
                except Exception:
                    pass

            t = Thread(target=_reader, daemon=True)
            t.start()
        except Exception:
            pass

    def _ffmpeg_read_frame(self):
        proc = getattr(self, '_ffmpeg_proc', None)
        if proc is None or proc.stdout is None:
            return None

        n = getattr(self, '_ffmpeg_frame_bytes', None)
        if not n:
            return None

        try:
            raw = proc.stdout.read(n)
        except Exception:
            return None

        if raw is None or len(raw) != n:
            return None

        h = int(self._ffmpeg_height)
        w = int(self._ffmpeg_width)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        return frame

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

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    configs = {
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 0,              # 0=norotation 
    }

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("RTPCapture")
   
    logger.log(logging.DEBUG, "Starting Capture")

    camera = rtpCapture(
        configs,
        port=554, 
        gpu = False)
    
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")

    window_handle = cv2.namedWindow("RTP Camera", cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty("RTP Camera", 0) >= 0):
        try:
            (frame_time, frame) = camera.capture.get()
            cv2.imshow('RTP Camera', frame)
        except: pass

        if cv2.waitKey(1) & 0xFF == ord('q'):  break

        try: 
            (level, msg)=camera.log.get_nowait()
            logger.log(level, "RTPCap:{}".format(msg))
        except: pass

    camera.stop()
    cv2.destroyAllWindows()
