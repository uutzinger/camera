###############################################################################
# Storage array data streamer
# Urs Utzinger 2020
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock
from threading import Event

#
import logging
import time
import tifffile


###############################################################################
# Disk Storage Server
###############################################################################

###############################################################################
# tiff data format
###############################################################################

class tiffServer(Thread):
    """Tiff file save """

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename):
        # initialize logger 
        self.logger = logging.getLogger("tiffStorageServer")

        # Threading Locks, Events
        self.stopped = True
        self.framearray_lock = Lock()

        # Initialize TIFF
        if filename is not None:
            self.tiff = tifffile.TiffWriter(filename, bigtiff=True)
        else:
            self.logger.log(logging.ERROR, "Status:Need to provide filename to store data!")
            return False

        # Init Frame and Thread
        self.framearray = None
        self.new_framearray  = False
        self.measured_cps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.tiff.close()
        self.stopped = True

    def start(self, storage_queue = None):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=(storage_queue,))
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self, storage_queue):
        """ run the thread """
        last_cps_time = time.time()
        num_cubes = 0
        while not self.stopped:
            # Storage througput calculation
            current_time = time.time()
            if (current_time - last_cps_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_cubes/5.0
                self.logger.log(logging.INFO, "Status:CPS:{}".format(self.measured_cps))
                num_cubes = 0
                last_cps_time = current_time

            if storage_queue is not None:
                if not storage_queue.empty(): 
                    (cube_time, data_cube) = storage_queue.get(block=True, timeout=None)
                    self.tiff.write(data_cube, compression='PACKBITS', photometric='MINISBLACK', contiguous=False, metadata ={'time': cube_time, 'author': 'camera'} )
                    # compression = 'LZW', 'LZMA', 'ZSTD', 'JPEG', 'PACKBITS', 'NONE', 'LERC'
                    # compression ='jpeg', 'png', 'zlib'
                    num_cubes += 1
            else:
                if self.new_framearray: 
                    self.tiff.write(self.framearray, compression='LZW', photometric='MINISBLACK', contiguous=False, metadata ={'time': self.framearrayTime, 'author': 'camera'} )
                    # Compression = 'LZW', 'LZMA', 'ZSTD', 'JPEG', 'PACKBITS', 'NONE', 'LERC'
                    # compression ='jpeg', 'png', 'zlib'
                    num_cubes += 1
                # run this no more than 100 times per second
                delay_time = 0.01 - (time.time() - current_time)
                if delay_time > 0.0:
                    time.sleep(delay_time)
 
    #
    # Data handling routines
    ###################################################################

    @property
    def framearray(self):
        """ returns most recent storage array """
        with self.framearray_lock:
            self._new_frame_array = False
            return self._frame_array
    @framearray.setter
    def framearray(self, val):
        """ set new array content """
        with self.framearray_lock:
            self._frame_array =  val
            self._new_frame_array = True

    @property
    def new_framearray(self):
        """ check if new array available """
        with self.framearray_lock:
            return self._new_frame_array
    @new_framearray.setter
    def new_framearray(self, val):
        """ override wether new array is available """
        with self.framearray_lock:
            self._new_frame_array = val
