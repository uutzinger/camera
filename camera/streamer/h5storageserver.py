###############################################################################
# HD5F storage array data streamer
# 
# create Streamer with h5 = h5Server(filename)
# Start Streamer with h5.start()
# Place frames into h5 with h5.queue.put((frame_time, frame))
#
# Storage array data streamer
# Urs Utzinger 
# 
# Changes:
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

# HDF5
import h5py

###############################################################################
# HDF5 Storage Server
###############################################################################

class h5Server(Thread):
    """ 
    HDF5 file array saver
    """

    _STOP_ITEM = (None, None)

    # Initialize the storage Thread
    def __init__(self, filename=None, queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        self.hdf5 = None
        self.hdf5_open = False

        # Initialize HDF5
        if filename is None:
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, "HDF5:Need to provide filename to store data!"))
            return

        try:
            self.hdf5 = h5py.File(filename, 'w')
            self.hdf5_open = True
        except Exception as exc:
            self.hdf5 = None
            self.hdf5_open = False
            if not self.log.full():
                self.log.put_nowait((logging.ERROR, f"HDF5:Could not create HDF5: {exc}"))
            return

        # Init
        self.measured_cps = 0.0

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
        """Close the underlying HDF5 file (idempotent)."""
        try:
            hdf5 = getattr(self, 'hdf5', None)
            if hdf5 is not None:
                hdf5.close()
        except Exception:
            pass
        self.hdf5 = None
        self.hdf5_open = False

    def start(self):
        """set the thread start conditions"""
        if self.is_alive():
            return
        self.stopped = False

        if not getattr(self, 'hdf5_open', False):
            self.stopped = True
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "HDF5:Not started because file is not open"))
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
                    name = str(cube_time)
                    if self.hdf5 is not None and name in self.hdf5:
                        # Avoid name collisions (e.g. repeated timestamps)
                        suffix = 1
                        while f"{name}_{suffix}" in self.hdf5:
                            suffix += 1
                        name = f"{name}_{suffix}"

                    self.hdf5.create_dataset(name, data=data_cube)  # ~11ms typical
                    num_cubes += 1
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.ERROR, f"HDF5:Write failed: {exc}"))

                # Storage throughput calculation
                current_time = time.time()
                if (current_time - last_time) >= 5.0:  # cube rate every 5 secs
                    self.measured_cps = num_cubes / 5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, "HDF5:CPS:{}".format(self.measured_cps)))
                    last_time = current_time
                    num_cubes = 0
        finally:
            self.close()
 
###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    import numpy as np
    from datetime import datetime
    import cv2

    display_interval =  0.01
    height =540
    width = 720
    depth = 14
    cube = np.random.randint(0, 255, (depth, height, width), 'uint8') 

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("HDF5")
   
    # Setting up Storage
    now = datetime.now()
    filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
    hdf5 = h5Server("C:\\temp\\" + filename)
    logger.log(logging.DEBUG, "Starting HDF5 Server")
    hdf5.start()


    window_handle = cv2.namedWindow("HDF5", cv2.WINDOW_AUTOSIZE)
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

        while not hdf5.log.empty():
            (level, msg) = hdf5.log.get_nowait()
            logger.log(level, "HDF5:{}".format(msg))

        if (current_time - last_display) > display_interval:
            frame = cube[0,:,:].copy() 
            cv2.putText(frame,"Frame:{}".format(num_frame), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('HDF5', frame)
            num_frame += 1
            last_display = current_time

            key = cv2.waitKey(1) 
            if (key == 27) or (key & 0xFF == ord('q')): stop = True
            if cv2.getWindowProperty("HDF5", 0) < 0: stop = True

            if not hdf5.queue.full(): hdf5.queue.put_nowait((current_time, cube))

    hdf5.stop()
    cv2.destroyAllWindows()
