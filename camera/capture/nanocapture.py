###############################################################################
# CSI video capture on Jetson Nano using gstreamer
# Urs Utzinger, 
#
# 2025, Refactor to GI/appsink + low-latency pipeline
# 2021, Initialize
# 2020, Update Queue
# 2019, First release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

# System
import logging, time

# Array
import numpy as np

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
# Video Capture
###############################################################################

class nanoCapture(Thread):
    """
    This thread continually captures frames from a CSI camera on Jetson using
    GStreamer (nvarguscamerasrc by default).
    """

    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    def __init__(self, configs, 
        camera_num: int = 0, 
        res: tuple = None,    # width, height
        exposure: float = None,
        queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        self._configs = configs or {}
        # Prefer nvargus on Jetson, allow override like gCapture
        self._gst_source = self._configs.get('gst_source', self._configs.get('source', 'nvargus'))
        self._gst_source_str = self._configs.get('gst_source_str', self._configs.get('gst_source_pipeline', None))
        self._gst_source_props_str = self._configs.get('gst_source_props_str', None)

        # populate desired settings from configuration file or function call
        ####################################################################
        self._camera_num = camera_num
        if exposure is not None:
            self._exposure   = exposure  
        else: 
            self._exposure   = self._configs.get('exposure', -1)
        if res is not None:
            self._camera_res = res
        else: 
            self._camera_res = self._configs.get('camera_res', (1280, 720))
        self._capture_width  = self._camera_res[0] 
        self._capture_height = self._camera_res[1]
        self._output_res     = self._configs.get('output_res', (-1, -1))
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._framerate      = self._configs.get('fps', 30)
        self._flip_method    = self._configs.get('flip', 0)

        # Threading Queue
        self.capture         = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True
        self.cam_lock        = Lock()

        # open up the camera
        self.open_cam()

        # Init vars
        self.frame_time   = 0.0
        self.measured_fps = 0.0

    # Thread routines 
    # Start Stop and Update Thread
    ###################################################################
    def stop(self): 
        """stop the thread"""
        self.stopped = True

    def start(self, capture_queue = None):
        """start the capture thread"""
        self.stopped = False
        super().start()

    # Thread entrypoint
    def run(self):
        """thread entrypoint"""
        self.update()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_frames = 0
        while not self.stopped:
            current_time = time.time()

            # Get New Image
            if getattr(self, "appsink", None) is not None:
                # GI bindings vary: prefer try_pull_sample, fall back to signal emit
                try_pull = getattr(self.appsink, "try_pull_sample", None)
                if callable(try_pull):
                    sample = try_pull(int(250 * 1e6))  # 250ms
                else:
                    sample = self.appsink.emit("try-pull-sample", int(250 * 1e6))

                if sample is not None:
                    img = self._sample_to_bgr(sample)
                    if img is not None:
                        num_frames += 1
                        self.frame_time = int(current_time * 1000)
                        if not self.capture.full():
                            self.capture.put_nowait((current_time * 1000.0, img))
                        else:
                            if not self.log.full():
                                self.log.put_nowait((logging.WARNING, "NanoCap:Capture Queue is full!"))
                else:
                    # no sample available within timeout
                    pass
            else:
                time.sleep(0.01)

            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "NanoCap:FPS:{}".format(self.measured_fps)))
                last_time = current_time
                num_frames = 0

        self.close_cam()

    def gstreamer_pipeline(self,
            camera_num:     int   = 0,
            capture_width:  int   = 1920, 
            capture_height: int   = 1080,
            output_width:   int   = 1280, 
            output_height:  int   = 720,
            framerate:      float = 30., 
            exposure_time:  float = -1,    # microseconds
            flip_method:    int   = 0):

            """
            Create gstreamer pipeline string.
            """
            ###################################################################################
            # gstreamer Options 
            # Examples for IX279
            ###################################################################################

            ###################################################################################
            # nvarguscamerasrc
            ###################################################################################
            # name                : The name of the object
            #                     flags: readable, writable
            #                     String. Default: "nvarguscamerasrc0"
            # parent              : The parent of the object
            #                     flags: readable, writable
            #                     Object of type "GstObject"
            # *blocksize          : Size in bytes to read per buffer (-1 = default)
            #                     flags: readable, writable
            #                     Unsigned Integer. Range: 0 - 4294967295 Default: 4096 
            #        'blocksize=-1 ' 
            # *num-buffers        : Number of buffers to output before sending EOS (-1 = unlimited)
            #                     flags: readable, writable
            #                     Integer. Range: -1 - 2147483647 Default: -1 
            #        'num-buffers=-1 '
            # typefind            : Run typefind before negotiating (deprecated, non-functional)
            #                     flags: readable, writable, deprecated
            #                     Boolean. Default: false
            # do-timestamp        : Apply current stream time to buffers
            #                     flags: readable, writable
            #                     Boolean. Default: true
            # *silent              : Produce verbose output ?
            #                     flags: readable, writable
            #                     Boolean. Default: true
            # *timeout            : timeout to capture in seconds (Either specify timeout or num-buffers, not both)
            #                     flags: readable, writable
            #                     Unsigned Integer. Range: 0 - 2147483647 Default: 0 
            #        'timeout=0 ' 
            # *wbmode             : White balance affects the color temperature of the photo
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamWBMode" Default: 1, "auto"
            #                         (0): off              - GST_NVCAM_WB_MODE_OFF
            #                         (1): auto             - GST_NVCAM_WB_MODE_AUTO
            #                         (2): incandescent     - GST_NVCAM_WB_MODE_INCANDESCENT
            #                         (3): fluorescent      - GST_NVCAM_WB_MODE_FLUORESCENT
            #                         (4): warm-fluorescent - GST_NVCAM_WB_MODE_WARM_FLUORESCENT
            #                         (5): daylight         - GST_NVCAM_WB_MODE_DAYLIGHT
            #                         (6): cloudy-daylight  - GST_NVCAM_WB_MODE_CLOUDY_DAYLIGHT
            #                         (7): twilight         - GST_NVCAM_WB_MODE_TWILIGHT
            #                         (8): shade            - GST_NVCAM_WB_MODE_SHADE
            #                         (9): manual           - GST_NVCAM_WB_MODE_MANUAL
            # *saturation         : Property to adjust saturation value
            #                     flags: readable, writable
            #                     Float. Range:               0 -               2 Default:               1 
            #        'saturation=1 ' 
            # sensor-id           : Set the id of camera sensor to use. Default 0.
            #                     flags: readable, writable
            #                     Integer. Range: 0 - 255 Default: 0 
            # *sensor-mode         : Set the camera sensor mode to use. Default -1 (Select the best match)
            #                     flags: readable, writable
            #                     Integer. Range: -1 - 255 Default: -1 
            #                     # -1..255, IX279 
            #                     # 0 (3264x2464,21fps)
            #                     # 1 (3264x1848,28fps)
            #                     # 2 (1080p, 30fps)
            #                     # 3 (1640x1232 30fps)
            #                     # 4 (720p, 60fps)
            #                     # 5 (720p, 120fps)
            # total-sensor-modes  : Query the number of sensor modes available. Default 0
            #                     flags: readable
            #                     Integer. Range: 0 - 255 Default: 0 
            # exposuretimerange   : Property to adjust exposure time range in nanoseconds
            #         Use string with values of Exposure Time Range (low, high)
            #         in that order, to set the property.
            #         eg: exposuretimerange="34000 358733000"
            #                     flags: readable, writable
            #                     String. Default: null
            # *gainrange           : Property to adjust gain range
            #         Use string with values of Gain Time Range (low, high)
            #         in that order, to set the property.
            #         eg: gainrange="1 16"
            #                     flags: readable, writable
            #                     String. Default: null
            #        'gainrange="1.0 10.625" '
            # * ispdigitalgainrange : Property to adjust digital gain range
            #         Use string with values of ISP Digital Gain Range (low, high)
            #         in that order, to set the property.
            #         eg: ispdigitalgainrange="1 8"
            #                     flags: readable, writable
            #                     String. Default: null
            #        'ispdigitalgainrange="1 8" '               
            # *tnr-strength        : property to adjust temporal noise reduction strength
            #                     flags: readable, writable
            #                     Float. Range:              -1 -               1 Default:              -1 
            #        'tnr-strength=-1 '
            # *tnr-mode            : property to select temporal noise reduction mode
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamTNRMode" Default: 1, "NoiseReduction_Fast"
            #                         (0): NoiseReduction_Off - GST_NVCAM_NR_OFF
            #                         (1): NoiseReduction_Fast - GST_NVCAM_NR_FAST
            #                         (2): NoiseReduction_HighQuality - GST_NVCAM_NR_HIGHQUALITY
            #        'tnr-mode=1 '
            # *ee-mode             : property to select edge enhnacement mode
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamEEMode" Default: 1, "EdgeEnhancement_Fast"
            #                         (0): EdgeEnhancement_Off - GST_NVCAM_EE_OFF
            #                         (1): EdgeEnhancement_Fast - GST_NVCAM_EE_FAST
            #                         (2): EdgeEnhancement_HighQuality - GST_NVCAM_EE_HIGHQUALITY
            #        #'ee-mode=0' 
            # *ee-strength         : property to adjust edge enhancement strength
            #                     flags: readable, writable
            #                     Float. Range:              -1 -               1 Default:              -1 
            #        #'ee-strength=-1 '
            # *aeantibanding       : property to set the auto exposure antibanding mode
            #                     flags: readable, writable
            #                     Enum "GstNvArgusCamAeAntiBandingMode" Default: 1, "AeAntibandingMode_Auto"
            #                         (0): AeAntibandingMode_Off - GST_NVCAM_AEANTIBANDING_OFF
            #                         (1): AeAntibandingMode_Auto - GST_NVCAM_AEANTIBANDING_AUTO
            #                         (2): AeAntibandingMode_50HZ - GST_NVCAM_AEANTIBANDING_50HZ
            #                         (3): AeAntibandingMode_60HZ - GST_NVCAM_AEANTIBANDING_60HZ
            #        'aeantibanding=1 '
            # *exposurecompensation: property to adjust exposure compensation
            #                     flags: readable, writable
            #                     Float. Range:              -2 -               2 Default:               0 
            #        'exposurecompensation=0 '                  # -2..2
            # *aelock              : set or unset the auto exposure lock
            #                     flags: readable, writable
            #                     Boolean. Default: false
            #        'aelock=true '
            # *awblock             : set or unset the auto white balance lock
            #                     flags: readable, writable
            #                     Boolean. Default: false
            #        'awblock=false ' 
            # *bufapi-version      : set to use new Buffer API
            #                     flags: readable, writable
            #                     Boolean. Default: false
            #        'bufapi-version=false '
            ###################################################################################

            # NOTE: We keep the actual pipeline minimal and feature-detect optional
            # properties to avoid pipeline parse failures across JetPack versions.
            #
            # If you want to pass additional nvarguscamerasrc properties (tnr-mode,
            # wbmode, etc.), provide them via configs['gst_source_props_str'].

            # deal with auto resizing
            if output_height <= 0:
                output_height = capture_height
            if output_width <= 0:
                output_width = capture_width

            gsrc_str = self._gstreamer_source(
                camera_num=camera_num,
                capture_width=capture_width,
                capture_height=capture_height,
                framerate=framerate,
                exposure_time=exposure_time,
            )

            # Map repo-wide flip enum (cv2Capture) to videoflip method
            if   flip_method == 0: flip = 0
            elif flip_method == 1: flip = 3  # ccw 90
            elif flip_method == 2: flip = 2  # 180
            elif flip_method == 3: flip = 1  # cw 90
            elif flip_method == 4: flip = 4  # horizontal
            elif flip_method == 5: flip = 7  # upright diagonal
            elif flip_method == 6: flip = 5  # vertical
            elif flip_method == 7: flip = 6  # upperleft diagonal
            else:                  flip = 0

            # Downstream pipeline (optimized like gCapture)
            parts = []

            if (output_width != capture_width) or (output_height != capture_height):
                parts.append('! videoscale ')
                parts.append('! video/x-raw, width=(int){:d}, height=(int){:d} '.format(int(output_width), int(output_height)))

            if flip != 0:
                parts.append('! videoflip method={:d} '.format(int(flip)))

            parts.append('! videoconvert ')
            parts.append('! video/x-raw, format=(string)BGR ')

            parts.append('! queue leaky=downstream max-size-buffers=1 ')
            parts.append('! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true enable-last-sample=false qos=false ')

            return gsrc_str + ''.join(parts)

            # Legacy pipeline example (kept for reference/documentation):
            # The original implementation set many nvarguscamerasrc tuning properties.
            # Prefer passing these via configs['gst_source_props_str'] so we only include
            # properties actually supported by the installed element.
            #
            # Example auto exposure range (ns): exposuretimerange="34000 358733000"
            # Example manual exposure range (ns): exposuretimerange="<x> <x>" + aelock=true

    def _gst_has_element(self, element_name: str) -> bool:
        if Gst is None:
            return False
        if not hasattr(self.__class__, "_gst_inited"):
            Gst.init(None)
            self.__class__._gst_inited = True
        try:
            return Gst.ElementFactory.find(element_name) is not None
        except Exception:
            return False

    def _gst_element_has_property(self, element_name: str, prop_name: str) -> bool:
        if Gst is None:
            return False
        if not hasattr(self.__class__, "_gst_inited"):
            Gst.init(None)
            self.__class__._gst_inited = True
        try:
            factory = Gst.ElementFactory.find(element_name)
            if factory is None:
                return False
            element = factory.create(None)
            if element is None:
                return False
            return element.find_property(prop_name) is not None
        except Exception:
            return False

    def _gstreamer_source(self,
        camera_num: int,
        capture_width: int,
        capture_height: int,
        framerate: float,
        exposure_time: float,
    ) -> str:
        """Build source portion similar to gCapture, defaulting to nvargus."""

        custom = getattr(self, "_gst_source_str", None)
        if isinstance(custom, str) and custom.strip():
            return custom.strip() + ' '

        extra_props = getattr(self, "_gst_source_props_str", None)
        extra_props_str = (extra_props.strip() + ' ') if isinstance(extra_props, str) and extra_props.strip() else ''

        source = getattr(self, "_gst_source", None)
        source = (source or 'nvargus').strip().lower()

        if source == 'auto':
            if self._gst_has_element('nvarguscamerasrc'):
                source = 'nvargus'
            elif self._gst_has_element('libcamerasrc'):
                source = 'libcamera'
            else:
                source = 'v4l2'

        if source in ('nvargus', 'nvarguscamerasrc'):
            # Best-effort properties: add only if supported by the element.
            # Keep these minimal to reduce JetPack version sensitivity.
            base_props = ''
            if self._gst_element_has_property('nvarguscamerasrc', 'do-timestamp'):
                base_props += 'do-timestamp=true '
            if self._gst_element_has_property('nvarguscamerasrc', 'timeout'):
                base_props += 'timeout=0 '
            if self._gst_element_has_property('nvarguscamerasrc', 'num-buffers'):
                base_props += 'num-buffers=-1 '
            if self._gst_element_has_property('nvarguscamerasrc', 'blocksize'):
                base_props += 'blocksize=-1 '

            exposure_props = ''
            # exposure_time is microseconds in this repo.
            if exposure_time is not None and exposure_time > 0:
                # Manual exposure: lock AE (if supported) and clamp exposuretimerange to a single value.
                exposure_ns = int(exposure_time * 1000)
                if self._gst_element_has_property('nvarguscamerasrc', 'exposuretimerange'):
                    exposure_props += 'exposuretimerange="{0} {0}" '.format(exposure_ns)
                if self._gst_element_has_property('nvarguscamerasrc', 'aelock'):
                    exposure_props += 'aelock=true '
            else:
                # Auto exposure: allow AE and provide a broad default exposure range if supported.
                if self._gst_element_has_property('nvarguscamerasrc', 'aelock'):
                    exposure_props += 'aelock=false '
                if self._gst_element_has_property('nvarguscamerasrc', 'exposuretimerange'):
                    # Legacy default range used in this repo (nanoseconds): 34us..358.733ms
                    exposure_props += 'exposuretimerange="34000 358733000" '

            return (
                'nvarguscamerasrc name=src sensor-id={:d} '.format(int(camera_num))
                + base_props
                + exposure_props
                + extra_props_str
                + '! video/x-raw(memory:NVMM), '
                + 'width=(int){:d}, height=(int){:d}, '.format(int(capture_width), int(capture_height))
                + 'framerate=(fraction){:d}/1, format=NV12 '.format(int(framerate))
                + '! nvvidconv '
                + '! video/x-raw, format=NV12 '
            )

        if source in ('libcamera', 'libcamerasrc'):
            return (
                'libcamerasrc name=src '.format(int(camera_num))
                + extra_props_str
                + '! video/x-raw, '
                + 'width=(int){:d}, height=(int){:d}, '.format(int(capture_width), int(capture_height))
                + 'framerate=(fraction){:d}/1, '.format(int(framerate))
                + 'format=NV12 '
            )

        if source in ('v4l2', 'v4l2src'):
            device = self._configs.get('device') or self._configs.get('v4l2_device')
            if not device:
                device = f'/dev/video{camera_num}'
            return (
                'v4l2src name=src device={:s} io-mode=2 '.format(device)
                + extra_props_str
                + '! video/x-raw, '
                + 'width=(int){:d}, height=(int){:d}, '.format(int(capture_width), int(capture_height))
                + 'framerate=(fraction){:d}/1 '.format(int(framerate))
            )

        return source + ' '

    #
    # Setup the Camera
    ############################################################################
    def open_cam(self):
        """Open up the camera so we can begin capturing frames."""

        self.gst = self.gstreamer_pipeline(
            camera_num     = self._camera_num,
            capture_width  = self._capture_width,
            capture_height = self._capture_height,
            output_width   = self._output_width,
            output_height  = self._output_height,
            framerate      = self._framerate,
            exposure_time  = self._exposure,
            flip_method    = self._flip_method,
        )

        if not self.log.full():
            self.log.put_nowait((logging.INFO, self.gst))

        if Gst is None:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "NanoCap:GStreamer python bindings (PyGObject) not available"))
            self.pipeline = None
            self.appsink = None
            self.source = None
            self.cam_open = False
            return

        if not hasattr(self.__class__, "_gst_inited"):
            Gst.init(None)
            self.__class__._gst_inited = True

        try:
            self.pipeline = Gst.parse_launch(self.gst)
            self.appsink = self.pipeline.get_by_name("appsink")
            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")

            # Optional: grab source for dynamic properties
            try:
                self.source = self.pipeline.get_by_name("src")
            except Exception:
                self.source = None

            # Ensure low-latency behavior even if the pipeline string is edited
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

            if not self.cam_open and not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "NanoCap:Failed to set pipeline to PLAYING"))
        except Exception as exc:
            self.pipeline = None
            self.appsink = None
            self.source = None
            self.cam_open = False
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"NanoCap:Failed to open GStreamer pipeline: {exc}"))

    def close_cam(self):
        if getattr(self, "pipeline", None) is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None
        self.source = None

    def _sample_to_bgr(self, sample):
        """Convert a Gst.Sample (video/x-raw,format=BGR) into a NumPy array."""
        if sample is None or GstVideo is None:
            return None

        caps = sample.get_caps()
        if caps is None:
            return None

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

    # Camera Routines
    ##################################################################

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, val):
        if val is None or val == -1:
            return
        self._exposure = float(val)

        # Best-effort live update on nvarguscamerasrc
        src = getattr(self, "source", None)
        if src is None:
            return

        try:
            exposure_ns = int(float(val) * 1000)
            with self.cam_lock:
                if src.find_property("exposuretimerange") is not None:
                    src.set_property("exposuretimerange", f"{exposure_ns} {exposure_ns}")
                if src.find_property("aelock") is not None:
                    src.set_property("aelock", True)
            if not self.log.full():
                self.log.put_nowait((logging.INFO, f"NanoCap:Exposure set:{val}"))
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, f"NanoCap:Failed to set exposure:{val}: {exc}"))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    import cv2

    configs = {
        'camera_res'      : (1280, 720),    # width & height
        'exposure'        : -1,             # microseconds, internally converted to nano seconds, <= 0 autoexposure
        'fps'             : 60,             # can not get more than 60fps
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 6,              # 0=norotation 
                                            # 1=ccw90deg 
                                            # 2=rotation180 
                                            # 3=cw90 
                                            # 4=horizontal 
                                            # 5=upright diagonal flip 
                                            # 6=vertical 
                                            # 7=uperleft diagonal flip
        'displayfps'       : 30
    }

    if configs['displayfps'] >= configs['fps']:  display_interval = 0
    else:                                        display_interval = 1.0/configs['displayfps']
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Nano Capture")

    logger.log(logging.DEBUG, "Starting Capture")

    camera = nanoCapture(configs, camera_num=0)
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")

    window_handle = cv2.namedWindow("Nano CSI Camera", cv2.WINDOW_AUTOSIZE)

    last_display = time.perf_counter()

    stop = False
    try:
        while not stop:
            current_time = time.perf_counter()

            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "NanoCap:{}".format(msg))

            try:
                (frame_time, frame) = camera.capture.get(timeout=0.25)
            except Exception:
                continue

            if (current_time - last_display) >= display_interval:
                cv2.imshow('Nano CSI Camera', frame)
                last_display = current_time
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop = True
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        try:
            camera.join(timeout=2.0)
        except Exception:
            pass
        try:
            camera.close_cam()
        except Exception:
            pass
        cv2.destroyAllWindows()