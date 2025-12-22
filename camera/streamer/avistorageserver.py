###############################################################################
# AVI storage array data streamer
#
# Create Streamer with avi = aviServer(filename, fps, size)
# Start Streamer with avi.start()
# Place frames into avi with avi.queue.put((frame_time, frame))
#
# Urs Utzinger 
#
# Changes:
# 2025 Codereview and cleanup
# 2021 Initial release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue
from queue import Empty

# System
import logging, time
import threading

# OpenCV
import cv2

###############################################################################
# Disk Storage Server: AVI
###############################################################################

class aviServer(Thread):
    """
    Save b/w and color images into avi-mjpg file
    """

    _STOP_ITEM = (None, None)

    # Initialize the storage thread
    def __init__(self, filename, fps, size, queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # Init vars
        self.measured_cps = 0.0

        # Initialize AVI
        self.avi = None
        self.avi_open = False

        if filename is None:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, "AVI:Need to provide filename to store avi!"))
            return

        try:
            self.avi = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
            self.avi_open = bool(self.avi) and self.avi.isOpened()
            if not self.avi_open:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"AVI:VideoWriter failed to open: {filename}"))
        except Exception as exc:
            self.avi = None
            self.avi_open = False
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"AVI:Could not create AVI: {exc}"))

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """Stop the thread (idempotent)."""
        self.stopped = True
        # Unblock writer thread if it's waiting on queue.get()
        try:
            if not self.queue.full():
                self.queue.put_nowait(self._STOP_ITEM)
        except Exception:
            pass

        try:
            if self.is_alive() and threading.current_thread() is not self:
                self.join(timeout=2.0)
        except Exception:
            pass

        self.close()

    def close(self):
        """Release underlying VideoWriter (idempotent)."""
        try:
            avi = getattr(self, 'avi', None)
            if avi is not None:
                avi.release()
        except Exception:
            pass
        self.avi = None
        self.avi_open = False

    def start(self):
        """set the thread start conditions"""
        if self.is_alive():
            return
        self.stopped = False

        # If VideoWriter could not be opened, don't start a busy thread.
        if not getattr(self, 'avi_open', False):
            self.stopped = True
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "AVI:Not started because VideoWriter is not open"))
            return

        super().start()

    # Thread entrypoint
    def run(self):
        self.update()

    # After Stating of the Thread, this runs continuously
    def update(self):
        """ run the thread """
        last_time = time.time()
        num_frames = 0

        try:
            while not self.stopped:
                try:
                    (frame_time, frame) = self.queue.get(timeout=0.25)
                except Empty:
                    continue

                if (frame_time, frame) == self._STOP_ITEM:
                    break

                if frame is None:
                    continue

                try:
                    self.avi.write(frame)
                    num_frames += 1
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"AVI:Write failed: {exc}"))

                # Storage throughput calculation
                current_time = time.time()
                if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                    self.measured_cps = num_frames/5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "AVI:FPS:{}".format(self.measured_cps)))
                    last_time = current_time
                    num_frames = 0
        finally:
            self.close()

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    import numpy as np
    from datetime import datetime

    fps              = 30
    res              = (1920,1080)
    height           = res[1]
    width            = res[0]
    depth            = 3
    display_interval =  1./fps

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("AVI")
   
    # Setting up Storage
    now = datetime.now()
    filename = now.strftime("%Y%m%d%H%M%S") + ".avi"
    avi = aviServer("C:\\temp\\" + filename, fps, res)
    logger.log(logging.DEBUG, "Starting AVI Server")
    avi.start()

    # synthetic image
    img = np.random.randint(0, 255, (height, width, depth), 'uint8') 

    window_handle = cv2.namedWindow("AVI", cv2.WINDOW_AUTOSIZE)
    font          = cv2.FONT_HERSHEY_SIMPLEX
    textLocation  = (10,20)
    fontScale     = 1
    fontColor     = (255,255,255)
    lineType      = 2

    last_display = time.time()
    num_frame = 0
    
    stop = False
    while(not stop):
        current_time = time.time()

        while not avi.log.empty():
            (level, msg) = avi.log.get_nowait()
            logger.log(level, "AVI:{}".format(msg))

        if (current_time - last_display) > display_interval:
            frame = img.copy()
            cv2.putText(frame,"Frame:{}".format(num_frame), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('AVI', frame)
            num_frame += 1
            last_display = current_time
            key = cv2.waitKey(1) 
            if (key == 27) or (key & 0xFF == ord('q')): stop = True
            if cv2.getWindowProperty("AVI", 0) < 0: stop = True

            if not avi.queue.full(): avi.queue.put_nowait((current_time, frame))

    avi.stop()
    cv2.destroyAllWindows()
