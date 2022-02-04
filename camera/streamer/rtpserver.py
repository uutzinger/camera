 
###############################################################################
# RTP point to point server
# 2021 Initial Release
# Urs Utzinger 
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

# Open Computer Vision
import cv2

###############################################################################
# RTP Server
###############################################################################

class rtpServer(Thread):
    """
    Create rtp h264 video network stream
    """

    # Initialize the RTP Thread
    def __init__(self, 
        resolution: tuple(int, int) = (320, 240), 
        fps: int     = 16,
        host: str    = '127.0.0.1', 
        port: int    = 554,  
        bitrate: int = 2048, 
        gpu: bool    = False):

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # populate desired settings from configuration file or function call
        self._port    = port
        self._host    = host
        self._res     = resolution
        self._fps     = fps
        self._fourcc  = 0
        self._bitrate = bitrate
        self._isColor = True
        self._gpuavail = gpu

        # https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/

        gst = 'appsrc ! videoconvert ! '
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano https://developer.nvidia.com/embedded/dlc/l4t-accelerated-gstreamer-guide-32-2 
                # control-rate 1 variable, 2 constant, 0 default
                # preset 0 ultrafast, 1 fast, 2 medium, 3 slow
                gst = ( gst + 
                    'omxh264enc control-rate=1 bitrate={:d} preset-level=1 ! '.format(self._bitrate*1000)   +
                    'video/x-h264,stream-format=(string)byte-stream ! h264parse ! ' )
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = ( gst + 
                    'omxh264enc control-rate=1 target-bitrate={:d} ! '.format(self._bitrate*1000)           +
                    'video/x-h264,stream-format=(string)byte-stream ! h264parse ! ' )
        else:
            if self._gpuavail:
                # preset 3 = lowlatency
                gst = gst + 'nvh264enc zerolatency=1 rc-mode=vbr max-bitrate={:d} ! '.format(self._bitrate) 
            else:
                gst = gst + 'x264enc tune=zerolatency bitrate={:d} speed-preset=superfast ! '.format(self._bitrate) 

        gst = gst + 'rtph264pay config-interval=1 pt=96 ! udpsink host={:s} port={:d}'.format(self._host, self._port)
        if not self.log.full(): self.log.put_nowait((logging.INFO, gst))

        self.rtp = cv2.VideoWriter(gst, apiPreference=cv2.CAP_GSTREAMER, fourcc=self._fourcc, fps=self._fps, frameSize=self._res, isColor=self._isColor)
        self.rtp_open = self.rtp.isOpened() 
        if not self.rtp_open:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "RTP:Failed to create rtp stream!"))
            return False
            
        self.measured_Cps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self):
        """set the thread start conditions"""
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After starting the thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_frames = 0
        
        while not self.stopped:
            if self.rtp is not None:
                (frame_time, frame) = self.queue.get(block=True, timeout=None)
                self.rtp.write(frame)
                num_frames += 1

            # RTP through put calculation
            current_time = time.time()
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_cps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "RTP:FPS:{}".format(self.measured_cps)))
                last_time = current_time
                num_frames = 0

        self.rtp.release()        


if __name__ == '__main__':

    import numpy as np

    fps=  30
    display_interval = 1./fps
    height =540
    width = 720
    depth = 3

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("RTP")
   
    # Setting up RTP
    rtp = rtpServer(
        resolution =(width,height),
        fps        = fps, 
        host       = '127.0.0.1', 
        port       = 554,
        bitrate    = 2048, 
        gpu        = False)
    logger.log(logging.DEBUG, "Starting RTP Server")
    rtp.start()

    # synthetic image
    cube = np.random.randint(0, 255, (height, width, depth), 'uint8') 

    last_display = time.time()
    num_frame = 0
    while True:
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            num_frame += 1
            last_display = current_time

            try:
                rtp.queue.put_nowait(cube)
            except: pass

            try: 
                (level, msg)=rtp.log.get_nowait()
                logger.log(level, "RTP:{}".format(msg))
            except: pass

    hdf5.stop()
    cv2.destroyAllWindows()
