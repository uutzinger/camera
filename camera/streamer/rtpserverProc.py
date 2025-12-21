 
###############################################################################
# RTP point to point server
# Attempt to run it with multiprocessing
#
# Public API:
#   rtp = rtpServerProc(...)
#   rtp.start()
#   rtp.queue.put_nowait(frame) OR rtp.queue.put_nowait((t, frame))
#   rtp.stop()
# Urs Utzinger
#
# Changes:
# 2025 Codereview and cleanup
# 2021 Initial Release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Processing
from multiprocessing import Process, Event, Queue
from queue import Empty

# System
import platform, time, logging

# Open Computer Vision
import cv2

###############################################################################
# RTP point to point Server
###############################################################################

class rtpServerProc(Process):
    """
    RTP h264 video network stream
    """

    # Initialize the RTP handler
    def __init__(self,
        resolution: tuple = (320, 240), 
        fps: int = 16, 
        host: str = '127.0.0.1', 
        port: int = 554,  
        bitrate: int = 2048, 
        color: bool = True,
        gpu: bool = False,
        queue_size: int = 32):

        super().__init__(daemon=True)

        self._STOP_ITEM = (None, None)

        # Process Locks, Events, Queue, Log
        self.stopper  = Event()
        self.log      = Queue(maxsize=32)
        self.queue    = Queue(maxsize=queue_size)

        # populate desired settings from the function call
        self._port     = port
        self._host     = host
        self._res      = resolution
        self._fps      = fps
        self._fourcc   = 0
        self._bitrate  = bitrate
        self._isColor  = color
        self._gpuavail = gpu

        # GSTREAMEER
        # https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/
        self.gst = 'appsrc ! videoconvert ! '
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': 
                # Jetson Nano https://developer.nvidia.com/embedded/dlc/l4t-accelerated-gstreamer-guide-32-2 
                # control-rate 1 variable, 2 constant, 0 default
                # preset 0 ultrafast, 1 fast, 2 medium, 3 slow
                self.gst = ( self.gst + 
                    'omxh264enc control-rate=1 bitrate={:d} preset-level=1 ! '.format(self._bitrate*1000)   +
                    'video/x-h264,stream-format=(string)byte-stream ! h264parse ! ' )
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                self.gst = ( self.gst + 
                    'omxh264enc control-rate=1 target-bitrate={:d} ! '.format(self._bitrate*1000)           +
                    'video/x-h264,stream-format=(string)byte-stream ! h264parse ! ' )
        else:
            if self._gpuavail:
                # could also use preset 3 for lowlatency
                self.gst = self.gst + 'nvh264enc zerolatency=1 rc-mode=vbr max-bitrate={:d} ! '.format(self._bitrate) 
            else:
                self.gst = self.gst + 'x264enc tune=zerolatency bitrate={:d} speed-preset=superfast ! '.format(self._bitrate) 

        self.gst = self.gst + 'rtph264pay config-interval=1 pt=96 ! udpsink host={:s} port={:d}'.format(self._host, self._port)
        if not self.log.full(): self.log.put_nowait((logging.INFO, self.gst))

    ###############################################################################
    # Process routines
    ###############################################################################

    def stop(self):
        """Stop the process (idempotent)."""
        self.stopper.set()
        try:
            if not self.queue.full():
                self.queue.put_nowait(self._STOP_ITEM)
        except Exception:
            pass

        try:
            if self.is_alive():
                self.join(timeout=2.0)
        except Exception:
            pass

        # Release OS resources associated with the process object (Python 3.7+)
        try:
            super().close()
        except Exception:
            pass

    def start(self):
        """Start the process (non-blocking)."""
        if self.is_alive():
            return

        self.stopper.clear()
        super().start()

    def close(self):
        """Alias for stop() to match other streamers."""
        self.stop()

    def run(self):
        """Child process body."""

        # Create rtp writer
        rtp = None
        try:
            rtp = cv2.VideoWriter(
                self.gst,
                apiPreference=cv2.CAP_GSTREAMER,
                fourcc=self._fourcc,
                fps=self._fps,
                frameSize=self._res,
                isColor=self._isColor,
            )
            if not rtp.isOpened():
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, "RTP:Failed to create rtp stream!"))
                return

            # Init
            last_time = time.time()
            num_frames = 0

            while not self.stopper.is_set():
                try:
                    item = self.queue.get(timeout=0.25)
                except Empty:
                    continue

                if item == self._STOP_ITEM:
                    break

                frame = None
                if isinstance(item, tuple) and len(item) == 2:
                    _, frame = item
                else:
                    frame = item

                if frame is None:
                    continue

                try:
                    rtp.write(frame)
                    num_frames += 1
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"RTP:Write failed: {exc}"))

                # RTP throughput calculation
                current_time = time.time()
                if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                    fps = num_frames/(current_time-last_time)
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "RTP:FPS:{}".format(fps)))
                    last_time = current_time
                    num_frames = 0
        finally:
            try:
                if rtp is not None:
                    rtp.release()
            except Exception:
                pass

