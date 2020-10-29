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

# 
import h5py
import tifffile

###############################################################################
# Disk Storage Server
###############################################################################

class h5Server(Thread):
    """ HDF5 file array saver    """

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename):
        # initialize logger 
        self.logger = logging.getLogger("h5StorageServer")

        # Threading Locks, Events
        self.framearray_lock = Lock() # When copying frame, lock it
        self.stopped = True

        # Initialize HDF5
        if filename is not None:
            self.f = h5py.File(filename,'w')
        else:
            self.logger.log(logging.ERROR, "Status:Need to provide filename to store data!")
            return False

        # Init Frame and Thread
        self.framearray = None
        self.new_framearray  = False
        #self.stopped    = False
        self.measured_aps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.f.close()
        self.stopped = True

    def start(self):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=())
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        last_time = time.time()
        last_aps_time = last_time
        num_arrays = 0
        while not self.stopped:
            current_time = time.time()
            # Storage througput calculation
            if (current_time - last_aps_time) >= 5.0: # framearray rate every 5 secs
                self.measured_aps = num_arrays/5.0
                self.logger.log(logging.DEBUG, "Status:APS:{}".format(self.measured_aps))
                num_arrays = 0
                last_aps_time = current_time
            if self.new_framearray: 
                with self.framearray_lock:
                    self.dset = self.f.create_dataset(str(self.framearrayTime), data=self.framearray) # 11ms
                    num_arrays += 1
            # run this no more than 500 times per second
            time.sleep(0.002)
 
    #
    # Data handling routines
    ###################################################################

    @property
    def framearray(self):
        """ returns most recent storage array """
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
        out = self._new_frame_array
        return out
    @new_framearray.setter
    def new_framearray(self, val):
        """ override wether new array is available """
        self._new_frame_array = val
        
    @property
    def framearrayTime(self):
        """ returns current array time """
        return self._frame_array_time
    @framearrayTime.setter
    def framearrayTime(self, val):
        """ set array time content """
        self._frame_array_time = val  

class tiffServer(Thread):
    """Tiff file save """

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename):
        # initialize logger 
        self.logger = logging.getLogger("tiffStorageServer")

        # Threading Locks, Events
        self.framearray_lock = Lock() # When copying frame, lock it
        self.stopped = True

        # Initialize HDF5
        if filename is not None:
            #self.f = h5py.File(filename,'w')
            self.tiff = TiffWriter(filename, bigtiff=True)
        else:
            self.logger.log(logging.ERROR, "Status:Need to provide filename to store data!")
            return False

        # Init Frame and Thread
        self.framearray = None
        self.new_framearray  = False
        #self.stopped    = False
        self.measured_aps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.tiff.close()
        self.stopped = True

    def start(self):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=())
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        last_time = time.time()
        last_aps_time = last_time
        num_arrays = 0
        while not self.stopped:
            current_time = time.time()
            # Storage througput calculation
            if (current_time - last_aps_time) >= 5.0: # framearray rate every 5 secs
                self.measured_aps = num_arrays/5.0
                self.logger.log(logging.DEBUG, "Status:APS:{}".format(self.measured_aps))
                num_arrays = 0
                last_aps_time = current_time
            if self.new_framearray: 
                with self.framearray_lock:
                    self.tiff.save(data=self.framearray, compress=6, photometric='minisblack', contiguous=False)
                    # We need to add tag: str(self.framearrayTime)
                    num_arrays += 1
            # run this no more than 500 times per second
            time.sleep(0.002)
 
    #
    # Data handling routines
    ###################################################################

    @property
    def framearray(self):
        """ returns most recent storage array """
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
        out = self._new_frame_array
        return out
    @new_framearray.setter
    def new_framearray(self, val):
        """ override wether new array is available """
        self._new_frame_array = val
        
    @property
    def framearrayTime(self):
        """ returns current array time """
        return self._frame_array_time
    @framearrayTime.setter
    def framearrayTime(self, val):
        """ set array time content """
        self._frame_array_time = val  
