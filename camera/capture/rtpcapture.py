###############################################################################
# RTP point to point video capture
# Uses opencv video capture to capture rtp stream
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
# 2021
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from queue import Queue
from threading import Thread, Lock

# System
import logging, time, platform

# Open Computer Vision
import cv2

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
    def __init__(self, configs,
        port: int          = 554,
        gpu: (bool)        = False,
        queue_size: int = 32):

        # populate settings
        #########################################################
        self._port           = port
        self._gpuavail       = gpu
        self._output_res     = configs['output_res']
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._flip_method    = configs['flip']

        # Threading Locks, Events
        self.log             = Queue(maxsize=queue_size)
        self.capture         = Queue(maxsize=32)
        self.stopped         = True
        self.cam_lock        = Lock()

        # open up the stream
        self._open_cam()

        # Init vars
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        Thread.__init__(self)


    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self, capture_queue = None):
        """set the thread start conditions"""
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_frames = 0
        while not self.stopped:
            current_time = time.time()

            _, img = self.capture.read()
            num_frames += 1
            self.frame_time = int(current_time*1000)

            if (img is not None) and (not self.capture.full()):

                if (self._output_height > 0) or (self._flip_method > 0):
                    # adjust output height
                    img_resized = cv2.resize(img, self._output_res)
                    # flip resized image
                    if   self._flip_method == 0: # no flipping
                        img_proc = img_resized
                    elif self._flip_method == 1: # ccw 90
                        img_proc = cv2.roate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 2: # rot 180, same as flip lr & up
                        img_proc = cv2.roate(img_resized, cv2.ROTATE_180)
                    elif self._flip_method == 3: # cw 90
                        img_proc = cv2.roate(img_resized, cv2.ROTATE_90_CLOCKWISE)
                    elif self._flip_method == 4: # horizontal
                        img_proc = cv2.flip(img_resized, 0)
                    elif self._flip_method == 5: # upright diagonal. ccw & lr
                        img_proc = cv2.flip(cv2.roate(img_resized, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                    elif self._flip_method == 6: # vertical
                        img_proc = cv2.flip(img_resized, 1)
                    elif self._flip_method == 7: # upperleft diagonal
                        img_proc = cv2.transpose(img_resized)
                    else:
                        img_proc = img_resized # not a valid flip method
                else:
                    img_proc = img

                self.capture.put_nowait((self.frame_time, img_proc))
            else:
                if not self.log.full(): self.log.put_nowait((logging.WARNING, "RTPCap:Capture Queue is full!"))

            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if not self.log.full(): self.logger.log(logging.INFO, "RTPCAP:FPS:{}".format(self.measured_fps))
                last_time = current_time
                num_frames = 0

        self.cam.release()        

    # Setup the Camera
    ############################################################################
    def _open_cam(self):
        """
        Open up the camera so we can begin capturing frames
        """

        ## Open the camera with platform optimal settings
        # https://answers.opencv.org/question/202017/how-to-use-gstreamer-pipeline-in-opencv/

        gst = 'udpsrc port={:d} caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtph264depay ! '.format(self._port)
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                gst = gst + 'h264parse ! omxh264dec ! nvvidconv ! '
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = gst + 'h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'decodebin ! videoconvert ! '
        gst = gst + 'appsink sync=false'
        self.logger.log(logging.INFO, gst)
        self.cam = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

        self.cam_open = self.cam.isOpened()        
        if not self.cam_open:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "RTPCap:Failed to open rtp stream!"))

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
