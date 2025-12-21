###############################################################################
# GStreamer camera capture, uses gstreamer to capture video
# 
# This is usually used for CSI video capture
#
# Urs Utzinger, 
#
# 2025, Updates
# 2022, First release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

# System
import logging, time
import platform

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

class gCapture(Thread):
    """
    This thread continually captures frames from gstreamer and adjusts camera 
    source depending the platform used.
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
        # Optional: choose source element without renaming the class
        self._gst_source = self._configs.get('gst_source', self._configs.get('source', None))
        # Optional: provide a full custom source string (must end before downstream videoconvert)
        self._gst_source_str = self._configs.get('gst_source_str', self._configs.get('gst_source_pipeline', None))
        # Optional: extra source-element properties, appended after element name (e.g. 'do-timestamp=true')
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
            self._camera_res = self._configs.get('camera_res', (640, 480))
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
            if self.appsink is not None:
                # GI bindings vary: prefer try_pull_sample, fall back to signal emit
                try_pull = getattr(self.appsink, "try_pull_sample", None)
                if callable(try_pull):
                    sample = try_pull(int(250 * 1e6))  # 250ms
                else:
                    sample = self.appsink.emit("try-pull-sample", int(250 * 1e6))
                if sample is not None:
                    img = self._sample_to_bgr(sample)
                    num_frames += 1
                    self.frame_time = int(current_time * 1000)

                    if (img is not None) and (not self.capture.full()):
                        self.capture.put_nowait((current_time * 1000.0, img))
                    else:
                        if not self.log.full():
                            self.log.put_nowait((logging.WARNING, "libcameraCap:Capture Queue is full!"))
                else:
                    # no sample available within timeout; loop again
                    pass

            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "libcameraCap:FPS:{}".format(self.measured_fps)))
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
            exposure_time:  float = -1,
            flip_method:    int   = 0):

            """     
            Create gstreamer pipeline string
            """

            # deal with auto resizing
            if output_height <= 0: output_height = capture_height
            if output_width  <= 0: output_width  = capture_width

            gsrc_str = self._gstreamer_source(
                camera_num=camera_num,
                capture_width=capture_width,
                capture_height=capture_height,
                framerate=framerate,
                exposure_time=exposure_time,
            )

            ###################################################################################
            # vide0/x-raw
            # https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html
            # videoscale
            #  width, height
            # videoflip
            #  method = 
            #   none (0) – Identity (no rotation)
            #   clockwise (1) – Rotate clockwise 90 degrees
            #   rotate-180 (2) – Rotate 180 degrees
            #   counterclockwise (3) – Rotate counter-clockwise 90 degrees
            #   horizontal-flip (4) – Flip horizontally
            #   vertical-flip (5) – Flip vertically
            #   upper-left-diagonal (6) – Flip across upper left/lower right diagonal
            #   upper-right-diagonal (7) – Flip across upper right/lower left diagonal
            #   automatic (8) – Select flip method based on image-orientation tag
            ###################################################################################

            if   flip_method == 0: flip = 0 # no flipping
            elif flip_method == 1: flip = 3 # ccw 90
            elif flip_method == 2: flip = 2 # rot 180, same as flip lr & up
            elif flip_method == 3: flip = 1 # cw 90
            elif flip_method == 4: flip = 4 # horizontal
            elif flip_method == 5: flip = 7 # upright diagonal. ccw & lr
            elif flip_method == 6: flip = 5 # vertical
            elif flip_method == 7: flip = 6 # upperleft diagonal

            # Downstream pipeline: keep it minimal
            # - Only insert videoscale if output size differs
            # - Only insert videoflip if requested
            # - Convert to BGR once right before appsink
            parts = []

            if (output_width != capture_width) or (output_height != capture_height):
                parts.append('! videoscale ')
                parts.append('! video/x-raw, width=(int){:d}, height=(int){:d} '.format(output_width, output_height))

            if flip != 0:
                parts.append('! videoflip method={:d} '.format(flip))

            parts.append('! videoconvert ')
            parts.append('! video/x-raw, format=(string)BGR ')

            # Latency controls:
            # - max-buffers=1 + drop=true keeps only the newest frame
            # - sync=false avoids clock syncing delays
            # - enable-last-sample=false avoids holding onto an extra buffer
            # - qos=false avoids upstream QoS throttling
            parts.append('! queue leaky=downstream max-size-buffers=1 ')
            parts.append('! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true enable-last-sample=false qos=false ')

            gstreamer_str = ''.join(parts)

            return (gsrc_str + gstreamer_str )

    def _gst_has_element(self, element_name: str) -> bool:
        if Gst is None:
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
        if not hasattr(self.__class__, "_gst_inited"):
            Gst.init(None)
            self.__class__._gst_inited = True
        try:
            return Gst.ElementFactory.find(element_name) is not None
        except Exception:
            return False

    def _gstreamer_source(self,
        camera_num: int,
        capture_width: int,
        capture_height: int,
        framerate: float,
        exposure_time: float,
    ) -> str:
        """Build the GStreamer source portion (gsrc_str).

        Source selection priority:
        1) configs['gst_source_str'] (full custom source string)
        2) configs['gst_source'] or configs['source'] (one of: auto, libcamera, v4l2, nvargus)
        3) auto-detect based on available GStreamer elements

        The returned string should end in a raw video stream that can be consumed by
        videoconvert/videoscale downstream.
        """

        custom = getattr(self, "_gst_source_str", None)
        if isinstance(custom, str) and custom.strip():
            return custom.strip() + ' '

        extra_props = getattr(self, "_gst_source_props_str", None)
        extra_props_str = (extra_props.strip() + ' ') if isinstance(extra_props, str) and extra_props.strip() else ''

        source = getattr(self, "_gst_source", None)
        if not source:
            # pull from configs if present
            try:
                source = self._configs.get('gst_source') or self._configs.get('source')
            except Exception:
                source = None
        source = (source or 'auto').strip().lower()

        # Auto-detect by plugin availability first
        if source == 'auto':
            if self._gst_has_element('libcamerasrc'):
                source = 'libcamera'
            elif self._gst_has_element('nvarguscamerasrc'):
                source = 'nvargus'
            else:
                source = 'v4l2'

        if source in ('libcamera', 'libcamerasrc'):
            return (
                'libcamerasrc name=libcamerasrc{:d} '.format(camera_num)
                + extra_props_str
                + '! video/x-raw, '
                + 'width=(int){:d}, height=(int){:d}, '.format(capture_width, capture_height)
                + 'framerate=(fraction){:d}/1, '.format(int(framerate))
                + 'format=NV12 '
            )

        if source in ('nvargus', 'nvarguscamerasrc'):
            # Jetson: nvarguscamerasrc produces NVMM memory; convert to system memory via nvvidconv
            # Note: property names depend on the installed JetPack/GStreamer build, so we only
            # include them if present to avoid pipeline parse failures.
            exposure_props = ''
            if exposure_time is not None and exposure_time > 0:
                # Treat exposure_time as microseconds (consistent with other modules in this repo)
                exposure_ns = int(exposure_time * 1000)
                if self._gst_element_has_property('nvarguscamerasrc', 'exposuretimerange'):
                    exposure_props += 'exposuretimerange=\"{0} {0}\" '.format(exposure_ns)
                # Lock AE if the element supports it (optional; safe-guarded)
                if self._gst_element_has_property('nvarguscamerasrc', 'aelock'):
                    exposure_props += 'aelock=true '

            return (
                'nvarguscamerasrc sensor-id={:d} '.format(camera_num)
                + exposure_props
                + extra_props_str
                + '! video/x-raw(memory:NVMM), '
                + 'width=(int){:d}, height=(int){:d}, '.format(capture_width, capture_height)
                + 'framerate=(fraction){:d}/1, format=NV12 '.format(int(framerate))
                + '! nvvidconv '
                + '! video/x-raw, format=NV12 '
            )

        if source in ('v4l2', 'v4l2src'):
            device = None
            try:
                device = self._configs.get('device') or self._configs.get('v4l2_device')
            except Exception:
                device = None
            if not device:
                device = f'/dev/video{camera_num}'
            return (
                'v4l2src device={:s} io-mode=2 '.format(device)
                + extra_props_str
                + '! video/x-raw, '
                + 'width=(int){:d}, height=(int){:d}, '.format(capture_width, capture_height)
                + 'framerate=(fraction){:d}/1 '.format(int(framerate))
            )

        # Fallback: treat as a literal GStreamer element/pipeline snippet
        return source + ' '
            
    #
    # Setup the Camera
    ############################################################################
    def open_cam(self):
        """
        Open up the camera so we can begin capturing frames
        """
        self.gst=self.gstreamer_pipeline(
            camera_num     = self._camera_num,
            capture_width  = self._capture_width,
            capture_height = self._capture_height,
            output_width   = self._output_width,
            output_height  = self._output_height,
            framerate      = self._framerate,
            exposure_time  = self._exposure,
            flip_method    = self._flip_method)

        if not self.log.full():
            self.log.put_nowait((logging.INFO, self.gst))

        if Gst is None:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "libcameraCap:GStreamer python bindings (PyGObject) not available"))
            self.pipeline = None
            self.appsink = None
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

            # Ensure low-latency behavior even if the pipeline string is edited
            try:
                self.appsink.set_property("sync", False)
                self.appsink.set_property("drop", True)
                self.appsink.set_property("max-buffers", 1)
                # BaseSink properties also available on appsink
                if self.appsink.find_property("enable-last-sample") is not None:
                    self.appsink.set_property("enable-last-sample", False)
                if self.appsink.find_property("qos") is not None:
                    self.appsink.set_property("qos", False)
            except Exception:
                pass

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            self.cam_open = ret != Gst.StateChangeReturn.FAILURE
        except Exception as exc:
            self.pipeline = None
            self.appsink = None
            self.cam_open = False
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"libcameraCap:Failed to open GStreamer pipeline: {exc}"))

    def close_cam(self):
        if getattr(self, "pipeline", None) is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None

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

            # Handle potential padding at row ends
            frame_2d = data[: height * stride].reshape((height, stride))
            frame = frame_2d[:, : width * 3].reshape((height, width, 3))
            return frame.copy()
        finally:
            buf.unmap(map_info)

    # Camera Routines
    ##################################################################

    # None yet

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    import cv2

    configs = {
        'camera_res'      : (3280, 2464),   # width & height
        'exposure'        : -1,             # microseconds, internally converted to nano seconds, <= 0 autoexposure
        'fps'             : 5,              # can not get more than 60fps
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 0,              # 0=norotation 
                                            # 1=ccw90deg 
                                            # 2=rotation180 
                                            # 3=cw90 
                                            # 4=horizontal 
                                            # 5=upright diagonal flip 
                                            # 6=vertical 
                                            # 7=uperleft diagonal flip
        'displayfps'       : 30
    }
    if configs['displayfps'] >= 0.8*configs['fps']:
        display_interval = 0
    else:
        display_interval = 1.0/configs['displayfps']
        
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Raspi libcamera Capture")

    logger.log(logging.DEBUG, "Starting Capture")
    
    camera = gCapture(configs, camera_num=0)
    camera.start()

    logger.log(logging.DEBUG, "Getting Frames")
    
    window_name = "Raspi libcamera CSI Camera"
    window_handle = cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    last_display = time.perf_counter()
    stop = False    

    while not stop:

        current_time = time.perf_counter()
        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if (current_time - last_display) >= display_interval:
            cv2.imshow(window_name, frame)
            last_display = current_time
            if cv2.waitKey(1) & 0xFF == ord('q'):  stop = True
            #try: 
            #    if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0: 
            #        stop = True
            #except: 
            #    stop = True
         
        while not camera.log.empty():
            (level, msg) = camera.log.get_nowait()
            logger.log(level, "{}".format(msg))

    camera.stop()
    cv2.destroyAllWindows()
