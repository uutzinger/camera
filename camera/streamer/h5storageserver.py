###############################################################################
# Storage array data streamer
# Urs Utzinger 2020
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue

# System
import logging, time

# HDF5
import h5py

###############################################################################
# HDF5 Storage Server
###############################################################################

class h5Server(Thread):
    """ 
    HDF5 file array saver
    """

    # Initialize the storage Thread
    def __init__(self, filename = None):

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # Initialize HDF5
        if filename is not None:
            try:
                self.hdf5 = h5py.File(filename,'w')
            except:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "HDF5:Could not create HDF5!"))
                return False
        else:
            if not self.log.full(): self.log.put_nowait((logging.ERROR, "HDF5:Need to provide filename to store data!"))
            return False

        # Init Frame and Thread
        self.measured_cps = 0.0

        Thread.__init__(self)

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self):
        """set the thread start conditions"""
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_cubes = 0

        while not self.stopped:
            (cube_time, data_cube) = self.queue.get(block=True, timeout=None) # This waits until item is available and removed from queue
            self.dset = self.hdf5.create_dataset(str(cube_time), data=data_cube) # 11ms
            num_cubes += 1

            # Storage througput calculation
            current_time = time.time()
            if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_cubes/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "HDF5:CPS:{}".format(self.measured_cps)))
                last_time = current_time
                num_cubes = 0

        self.hdf5.close()
 
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

    while(cv2.getWindowProperty("HDF5", 0) >= 0):
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            frame = cube[0,:,:].copy() 
            cv2.putText(frame,"Frame:{}".format(num_frame), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('HDF5', frame)
            num_frame += 1
            last_display = current_time

            key = cv2.waitKey(1) 
            if (key == 27) or (key & 0xFF == ord('q')): break

            try: hdf5.queue.put_nowait((current_time, cube))
            except: pass

        while not hdf5.log.empty():
            (level, msg)=hdf5.log.get_nowait()
            logger.log(level, "HDF5:{}".format(msg))

    hdf5.stop()
    cv2.destroyAllWindows()
