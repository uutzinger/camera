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
import h5py


###############################################################################
# Disk Storage Server
###############################################################################

###############################################################################
# h5 data format
###############################################################################

class h5Server(Thread):
    """ HDF5 file array saver    """

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename = None):
        # initialize logger 
        self.logger = logging.getLogger("h5StorageServer")

        # Threading Locks, Events
        self.stopped = True
        self.framearray_lock = Lock()

        # Initialize HDF5
        if filename is not None:
            self.f = h5py.File(filename,'w')
        else:
            self.logger.log(logging.ERROR, "Status:Need to provide filename to store data!")
            return False

        # Init Frame and Thread
        self.framearray = None
        self.new_framearray  = False
        self.framearray_time = 0.0
        self.measured_cps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True
        time.sleep(0.5) # give thread time to finish
        self.f.close()

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
                (cube_time, data_cube) = storage_queue.get(block=True, timeout=None) # This waits until item is available and removed from queue
                self.dset = self.f.create_dataset(str(cube_time), data=data_cube) # 11ms
                last_cube_time = cube_time
                num_cubes += 1
            else:
                if self.new_framearray: 
                    self.dset = self.f.create_dataset(str(self.framearray_time), data=self.framearray) # 11ms
                    num_cubes += 1
                # run this no more than 100 times per second
                loop_delay = time.time() - current_time
                if loop_delay < 0.01:
                    time.sleep((0.01-loop_delay))
                # dont sleep if saving already consumed 10ms
 
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
            self._frame_array = val
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
