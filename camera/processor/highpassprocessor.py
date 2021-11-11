###############################################################################
# Highpass Filter
# Urs Utzinger 2022
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
import numpy as np
from numba import vectorize

###############################################################################
# Basic Image Processor
###############################################################################
# Numpy Vectorized Image Processor
@vectorize(['float32(uint8, float32, float32)'], nopython=True, fastmath=True)
def movingavg(data, average, alpha):
    return np.add(np.multiply(average, 1.-alpha), np.multiply(data, alpha))

@vectorize(['float32(float32, uint8)'], nopython=True, fastmath=True)
def highpass(averageData, data):
    return np.subtract(averageData, data)

class highpassProcessor(Thread):
    """Background removal, flat field correction, white balance """

    # Initialize the Processor Thread
    def __init__(self, res: (int, int, int), alpha: float = 0.05 ):
        # initialize logger 
        self.logger = logging.getLogger("highpassProcessor")

        # Threading Locks, Events
        self.stopped = True
        self.framearray_lock = Lock()

        # Initialize Processor
        self.alpha = alpha
        self.averageData = np.zeros(res, 'float32')
        self.filteredData = np.zeros(res, 'float32') 

        # Init Frame and Thread
        self.measured_cps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.proc.close()
        self.stopped = True

    def start(self, input_queue, output_queue):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=(input_queue, output_queue))
        T.daemon = True # run in background
        T.start()

    # After Starting the Thread, this runs continously
    def update(self, input_queue, output_queue):
        """ run the thread """
        last_cps_time = time.time()
        num_cubes = 0
        while not self.stopped:
            # Processing throughput calculation
            current_time = time.time()
            if (current_time - last_cps_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_cubes/5.0
                self.logger.log(logging.INFO, "Status:CPS:{}".format(self.measured_cps))
                num_cubes = 0
                last_cps_time = current_time

            if not input_queue.empty():
                ############################################################################
                #
                # Image Processing
                #
                ############################################################################
                # obtain new data cube
                (data_time, data) = input_queue.get(block=True, timeout=None)
                # Process
                self.averageData  = movingavg(data, self.averageData, self.alpha)
                self.filteredData = highpass(self.averageData, data)
                # Put results into output queue
                if not output_queue.full():
                    output_queue.put((data_time, self.filteredData), block=False)
                else:
                    self.logger.log(logging.WARNING, "Status:Processed Output Queue is full!")
                num_cubes += 1
 