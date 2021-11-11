 
###############################################################################
# RTP point to point server
# Urs Utzinger
# 2021
#
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock
import platform

#
import logging
import time

# Open Computer Vision
import cv2

###############################################################################
# Video Server
###############################################################################

class rtpServer(Thread):
    """
    rtp h264 video network stream
    """

    # Initialize the RTP Thread
    # Opens Capture Device
    def __init__(self, resolution: (int, int) = (320, 240), fps: int = 16, host: str = '127.0.0.1', port: int = 554,  bitrate: int = 2048, GPU: bool = False):

        # initialize LOGGER
        self.logger = logging.getLogger("rtpServer")

        # Threading Locks, Events
        self.frame_lock = Lock()
        self.stopped    = True

        # populate desired settings from configuration file or function call
        self._port    = port
        self._host    = host
        self._res     = resolution
        self._fps     = fps
        self._fourcc  = 0
        self._bitrate = bitrate
        self._isColor = True
        self._gpuavail = GPU

        # https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/

        plat = platform.system()
        if plat == "Windows":
            if self._gpuavail:
                # preset 3 = lowlatency
                gst = (
                    'appsrc ! videoconvert ! '                                                              + 
                    'nvh264enc zerolatency=1 rc-mode=vbr max-bitrate={:d} ! '.format(self._bitrate) +
                    'rtph264pay config-interval=1 pt=96 ! '                                                 +
                    'udpsink host={:s} port={:d}'.format(self._host, self._port) )
            else:
                gst = (
                    'appsrc ! videoconvert ! '                                                              + 
                    'x264enc tune=zerolatency bitrate={:d} speed-preset=superfast ! '.format(self._bitrate) +
                    'rtph264pay config-interval=1 pt=96 ! '                                                 +
                    'udpsink host={:s} port={:d}'.format(self._host, self._port) )
        elif plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano https://developer.nvidia.com/embedded/dlc/l4t-accelerated-gstreamer-guide-32-2 
                gst = (
                    # control-rate 1 variable, 2 constant, 0 default
                    # preset 0 ultrafast, 1 fast, 2 medium, 3 slow
                    'appsrc ! videoconvert ! '                                                              + 
                    'omxh264enc control-rate=1 bitrate={:d} preset-level=1 ! '.format(self._bitrate*1000)   +
                    'video/x-h264,stream-format=(string)byte-stream ! h264parse ! '                         +
                    'rtph264pay config-interval=1 pt=96 ! '                                                 +
                    'udpsink host={:s} port={:d}'.format(self._host, self._port) )
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = (
                    'appsrc ! videoconvert ! '                                                              +
                    'omxh264enc control-rate=1 target-bitrate={:d} ! '.format(self._bitrate*1000)           +
                    'video/x-h264,stream-format=(string)byte-stream ! h264parse ! '                         +
                    'rtph264pay config-interval=1 pt=96 ! '                                                 +
                    'udpsink host={:s} port={:d}'.format(self._host, self._port) )
        elif plat == "MacOS":
            gst = (
                'appsrc ! videoconvert ! '                                                              + 
                'x264enc tune=zerolatency bitrate={:d} speed-preset=superfast ! '.format(self._bitrate) +
                'rtph264pay config-interval=1 pt=96 ! '                                                 +
                'udpsink host={:s} port={:d}'.format(self._host, self._port) )
        else:
            gst = (
                'appsrc ! videoconvert ! '                                                              + 
                'x264enc tune=zerolatency bitrate={:d} speed-preset=superfast ! '.format(self._bitrate) +
                'rtph264pay config-interval=1 pt=96 ! '                                                 +
                'udpsink host={:s} port={:d}'.format(self._host, self._port) )

        self.logger.log(logging.INFO, gst)

        self.rtp = cv2.VideoWriter(gst, apiPreference=cv2.CAP_GSTREAMER, fourcc=self._fourcc, fps=self._fps, frameSize=self._res, isColor=self._isColor)
        self.stream_open = self.rtp.isOpened() 
        if not self.stream_open:
            self.logger.log(logging.CRITICAL, "Status:Failed to create rtp stream!")

        # Init Frame and Thread
        self.frame     = None
        self.new_frame = False
        self.measured_fps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self, rtp_queue = None):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=(rtp_queue,))
        T.daemon = True # run in background
        T.start()

    # After starting the thread, this runs continously
    def update(self, rtp_queue):
        """ run the thread """
        last_fps_time = time.time()
        num_frames = 0
        
        while not self.stopped:
            # rtp through put calculation
            current_time = time.time()
            if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.logger.log(logging.INFO, "Status:FPS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time

            # take image from queue and sent to rtp
            if rtp_queue is not None:
                if not rtp_queue.empty(): 
                    (frame_time, frame) = rtp_queue.get(block=True, timeout=None)
                    self.rtp.write(frame)
                    num_frames += 1
            else:
                if self.new_frame: 
                    self.rtp.write(self.frame)
                    num_frames += 1
                # run this no more than 100 times per second
                delay_time = 0.01 - (time.time() - current_time)
                if delay_time > 0.0:
                    time.sleep(delay_time)

            if self.stopped:
                self.rtp.release()        

    #
    # Frame routines ##################################################
    # Each camera stores frames locally
    ###################################################################

    @property
    def frame(self):
        """ returns most recent frame """
        self._new_frame = False
        return self._frame

    @frame.setter
    def frame(self, img):
        """ set new frame content """
        with self.frame_lock:
            self._frame = img
            self._new_frame = True

    @property
    def new_frame(self):
        """ check if new frame available """
        out = self._new_frame
        return out

    @new_frame.setter
    def new_frame(self, val):
        """ override wether new frame is available """
        self._new_frame = val
