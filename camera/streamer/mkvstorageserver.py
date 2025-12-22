###############################################################################
# MKV storage array data streamer
# 
# create Streamer with mkv = mkvServer(filename, fps, size)
# Start Streamer with mkv.start()
# Place frames with mkv.queue.put((frame_time, frame))
#
# Urs Utzinger 
#
# Changes:
# 2025 Codereview and cleanup
# 2021 Initial Release
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
# MKV Disk Storage Server
###############################################################################

class mkvServer(Thread):
    """
    Save in  Matroska multimedia container format, mp4v
    """

    _STOP_ITEM = (None, None)

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename, fps, size, queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # Init vars
        self.measured_cps = 0.0

        # Initialize MKV
        self.mkv = None
        self.mkv_open = False

        if filename is None:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, "MKV:Need to provide filename to store mkv!"))
            return

        try:
            self.mkv = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            self.mkv_open = bool(self.mkv) and self.mkv.isOpened()
            if not self.mkv_open:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"MKV:VideoWriter failed to open: {filename}"))
        except Exception as exc:
            self.mkv = None
            self.mkv_open = False
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"MKV:Could not create MKV: {exc}"))

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

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
            mkv = getattr(self, 'mkv', None)
            if mkv is not None:
                mkv.release()
        except Exception:
            pass
        self.mkv = None
        self.mkv_open = False

    def start(self, storage_queue = None):
        """set the thread start conditions"""
        if self.is_alive():
            return
        self.stopped = False

        if not getattr(self, 'mkv_open', False):
            self.stopped = True
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "MKV:Not started because VideoWriter is not open"))
            return

        super().start()

    # Thread entrypoint
    def run(self):
        self.update()

    # After Starting of the Thread, this runs continuously
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
                    self.mkv.write(frame)
                    num_frames += 1
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"MKV:Write failed: {exc}"))

                # Storage throughput calculation
                current_time = time.time()
                if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                    self.measured_cps = num_frames/5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "MKV:FPS:{}".format(self.measured_cps)))
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
    stop = False
    while(not stop):
        current_time = time.time()

        while not mkv.log.empty():
            (level, msg)=mkv.log.get_nowait()
            logger.log(level, "MKV:{}".format(msg))

        if (current_time - last_display) > display_interval:
            frame = img.copy()
            cv2.putText(frame,"Frame:{}".format(num_frames), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('MKV', frame)
            num_frames += 1
            last_display = current_time
            key = cv2.waitKey(1)
            if (key == 27) or (key & 0xFF == ord('q')):
                stop = True
            try:
                if cv2.getWindowProperty("MKV", 0) < 0:
                    stop = True
            except Exception:
                stop = True

            if not mkv.queue.full(): mkv.queue.put_nowait((current_time, frame))

    mkv.stop()
    cv2.destroyAllWindows()
