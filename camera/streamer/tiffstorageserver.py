###############################################################################
# TIFF storage array data streamer
# 
# create Streamer with tiff = tiffServer(filename)
# Start Streamer with tiff.start()
# Place frames with tiff.queue.put((frame_time, frame))
#
# Storage array data streamer
# Urs Utzinger 
# 
# Chagnges:
# 2025 Codereview and cleanup
# 2020 Initial release
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

# TIFF
import tifffile


###############################################################################
# TIFF Storage Server
###############################################################################

class tiffServer(Thread):
    """
    Tiff saver
    """

    _STOP_ITEM = (None, None)

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename=None, queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        self.tiff = None
        self.tiff_open = False

        # Initialize TIFF
        if filename is None:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, "TIFF:Need to provide filename to store data!"))
            return

        try:
            self.tiff = tifffile.TiffWriter(filename, bigtiff=True)
            self.tiff_open = True
        except Exception as exc:
            self.tiff = None
            self.tiff_open = False
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"TIFF:Could not create TIFF: {exc}"))
            return

        self.measured_cps = 0.0

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
        """Close the underlying TiffWriter (idempotent)."""
        try:
            tiff = getattr(self, 'tiff', None)
            if tiff is not None:
                tiff.close()
        except Exception:
            pass
        self.tiff = None
        self.tiff_open = False

    def start(self):
        """ set the thread start conditions """
        if self.is_alive():
            return
        self.stopped = False

        if not getattr(self, 'tiff_open', False):
            self.stopped = True
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "TIFF:Not started because file is not open"))
            return

        super().start()

    # Thread entrypoint
    def run(self):
        self.update()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_cubes = 0

        try:
            while not self.stopped:
                try:
                    (cube_time, data_cube) = self.queue.get(timeout=0.25)
                except Empty:
                    continue

                if (cube_time, data_cube) == self._STOP_ITEM:
                    break

                if cube_time is None or data_cube is None:
                    continue

                try:
                    # compression = 'LZW', 'LZMA', 'ZSTD', 'JPEG', 'PACKBITS', 'NONE', 'LERC'
                    # compression = 'jpeg', 'png', 'zlib'
                    self.tiff.write(
                        data_cube,
                        compression='PACKBITS',
                        photometric='MINISBLACK',
                        contiguous=False,
                        metadata={'time': cube_time, 'author': 'camera'},
                    )
                    num_cubes += 1
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"TIFF:Write failed: {exc}"))

                # Storage throughput calculation
                current_time = time.time()
                if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                    self.measured_cps = num_cubes/5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "TIFF:CPS:{}".format(self.measured_cps)))
                    num_cubes = 0
                    last_time = current_time
        finally:
            self.close()

if __name__ == '__main__':

    import numpy as np
    from datetime import datetime
    import cv2

    display_interval =  0.01
    height =540
    width = 720
    depth = 3 # can display only 3 colors ;-)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("TIFF")
   
    # Setting up Storage
    now = datetime.now()
    filename = now.strftime("%Y%m%d%H%M%S") + ".tiff"
    tiff = tiffServer("C:\\temp\\" + filename)
    logger.log(logging.DEBUG, "Starting TIFF Server")
    tiff.start()

    # synthetic image
    image = np.random.randint(0, 255, (depth, height, width), 'uint8') 

    window_handle = cv2.namedWindow("TIFF", cv2.WINDOW_AUTOSIZE)
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

        while not tiff.log.empty():
            (level, msg)=tiff.log.get_nowait()
            logger.log(level, "TIFF:{}".format(msg))

        if (current_time - last_display) > display_interval:
            last_display = current_time
            frame = image[1,:,:].copy()
            cv2.putText(frame,"Frame:{}".format(num_frames), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('TIFF', frame)
            num_frames += 1
            key = cv2.waitKey(1) 
            if (key == 27) or (key & 0xFF == ord('q')): stop = True
            if cv2.getWindowProperty("TIFF", 0) < 0: stop = True

            if not tiff.queue.full(): tiff.queue.put_nowait((current_time, image))


    tiff.stop()
    cv2.destroyAllWindows()
