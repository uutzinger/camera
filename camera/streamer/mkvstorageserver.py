###############################################################################
# MKV storage array data streamer
# 2021 Initial Release
# Urs Utzinger 
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue

# System
import logging, time

# OpenCV
import cv2

###############################################################################
# MKV Disk Storage Server
###############################################################################

class mkvServer(Thread):
    """
    Save in  Matroska multimedia container format, mp4v
    """

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename, fps, size):

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # Init vars
        self.measured_cps = 0.0

        # Initialize MKV
        if filename is not None:
            try: 
                self.mkv = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            except:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "MKV:Could not create MKV!"))
                return False
        else:
            if not self.log.full(): self.log.put_nowait((logging.ERROR, "MKV:Need to provide filename to store mkv!"))
            return False

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self, storage_queue = None):
        """set the thread start conditions"""
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        last_time = time.time()
        num_frames = 0

        while not self.stopped:
            (frame_time, frame) = self.queue.get(block=True, timeout=None)
            self.mkv.write(frame)
            num_frames += 1

            # Storage througput calculation
            current_time = time.time()
            if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "MKV:FPS:{}".format(self.measured_cps)))
                last_time = current_time
                num_frames = 0

        self.mkv.release()
 
###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    import numpy as np
    from datetime import datetime

    fps              = 30
    res              = (720,540)
    height           = res[1]
    width            = res[0]
    display_interval =  1./fps

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("MKV")
   
    # Setting up Storage
    now = datetime.now()
    filename = now.strftime("%Y%m%d%H%M%S") + ".mkv"
    mkv = mkvServer("C:\\temp\\" + filename, fps, res)
    logger.log(logging.DEBUG, "Starting MKV Server")
    mkv.start()

    # synthetic image
    img = np.random.randint(0, 255, (height, width, 3), 'uint8') 

    window_handle = cv2.namedWindow("MKV", cv2.WINDOW_AUTOSIZE)
    font          = cv2.FONT_HERSHEY_SIMPLEX
    textLocation  = (10,20)
    fontScale     = 1
    fontColor     = (255,255,255)
    lineType      = 2

    last_display = time.time()
    num_frames = 0
    while(cv2.getWindowProperty("MKV", 0) >= 0):
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            frame = img.copy()
            cv2.putText(frame,"Frame:{}".format(num_frames), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('MKV', frame)
            num_frames += 1
            last_display = current_time
            key = cv2.waitKey(1) 
            if (key == 27) or (key & 0xFF == ord('q')): break

            try: mkv.queue.put_nowait((current_time, frame))
            except: pass

        while not mkv.log.empty():
            (level, msg)=mkv.log.get_nowait()
            logger.log(level, "MKV:{}".format(msg))

    mkv.stop()
    cv2.destroyAllWindows()
