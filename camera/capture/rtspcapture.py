 
###############################################################################
# RTSP video capture
# Uses opencv video capture to capture rtsp stream
# Adapts to operating system and allows configuation of codec
# Urs Utzinger
# 2021, Initialize, Remove frame access (use only queue)
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
        gpu: (bool) = False):
        
        # populate desired settings from configuration file or function call
        if rtsp is not None: 
            self._rtsp      = rtsp
        else: 
            self._rtsp      = configs['rtsp']
        self._output_res    = configs['output_res']
        self._output_width  = self._output_res[0]
        self._output_height = self._output_res[1]
        self._flip_method   = configs['flip']
        self._gpuavail      = gpu

        # Threading Locks, Events
        # Threading Queue, Locks, Events
        self.capture         = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # open up the camera
        self._open_cam()

        # Init Frame and Thread
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        Thread.__init__(self)

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        # clean up

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

            _, img = self.cam.read()
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
                self.log.put_nowait((logging.WARNING, "RTSPCap:Capture Queue is full!"))

            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.log.put_nowait((logging.INFO, "RTSPCAM:FPS:{}".format(self.measured_fps)))
                num_frames = 0
                last_time = current_time

        self.cam.release()        

    # Setup the Camera
    ############################################################################

    def _open_cam(self):
        """
        Open up the camera so we can begin capturing 
        """

        gst ='rtspsrc location=' + self._rtsp + ' latency=10 ! rtph264depay ! h264parse ! '
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64': # Jetson Nano
                gst = gst + 'omxh264dec ! nvvidconv ! appsink sync=false'
            elif platform.machine() == 'armv6l' or platform.machine() == 'armv7l': # Raspberry Pi
                gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'avdec_h264 ! videoconvert ! '
        gst = gst + 'appsink sync=false'

        self.log.put_nowait((logging.INFO, gst))
        self.cam = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)

        self.cam_open = self.cam.isOpened()        
        if not self.cam_open:
            self.log.put_nowait((logging.CRITICAL, "RTSPCap:Failed to open camera!"))

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
