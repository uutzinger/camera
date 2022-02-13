###############################################################################
# Highpass Filter
# Urs Utzinger 2022
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue

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
# y = (1-alpha) * y + alpha * x
@vectorize(['float32(uint16, float32, float32)'], nopython=True, fastmath=True)
def movingavg(data, average, alpha):
    return np.add(np.multiply(average, 1.-alpha), np.multiply(data, alpha))

@vectorize(['float32(uint16, float32)'], nopython=True, fastmath=True)
def highpass(data, average):
    return np.subtract(data, average)

class highpassProcessor(Thread):
    """Highpass filter"""

    # Initialize the Processor Thread
    def __init__(self, res: (int, int, int), alpha: float = 0.95 ):

        # Threading Locks, Events
        self.stopped = True

        # Initialize Processor
        self.alpha = alpha
        self.averageData  = np.zeros(res, 'float32')
        self.filteredData = np.zeros(res, 'float32') 

        # Init Frame and Thread
        self.input   = Queue(maxsize=32)
        self.output  = Queue(maxsize=32)
        self.log     = Queue(maxsize=32)
        self.stopped = True

        Thread.__init__(self)

    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self):
        """set the thread start conditions"""
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After Starting the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_cubes = 0
        total_time = 0
        while not self.stopped:

            ############################################################################
            # Image Processing
            ############################################################################
            # obtain new data
            (data_time, data) = self.input.get(block=True, timeout=None)
            # process
            start_time = time.perf_counter()
            self.averageData  = movingavg(data, self.averageData, self.alpha)
            self.filteredData = highpass(data, self.averageData)
            total_time += time.perf_counter() - start_time
            # put results into output queue
            if not self.output.full():
                self.output.put_nowait((data_time, self.filteredData, self.averageData))
            else:
               if not self.log.full(): self.log.put_nowait((logging.WARNING, "Proc:Processed Output Queue is full!"))
            num_cubes += 1
 
            ############################################################################
            # Processing throughput calculation
            ############################################################################
            current_time = time.time()
            if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_cubes/5.0
                self.measured_time = total_time/num_cubes
                if not self.log.full(): self.log.put_nowait((logging.INFO, "HighPass:CPS:{}".format(self.measured_cps)))
                if not self.log.full(): self.log.put_nowait((logging.INFO, "HighPass:Time:{}".format(self.measured_time)))
                num_cubes = 0
                total_time = 0
                last_time = current_time
