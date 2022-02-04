###############################################################################
# Highpass Filter
# Urs Utzinger 2022
# https://dsp.stackexchange.com/questions/12757/a-better-high-order-low-pass-filter
# https://www.dsprelated.com/freebooks/sasp/Running_Sum_Lowpass_Filter.html
# https://www.dsprelated.com/showarticle/1337.php
#
# Moving Average (D-1 additions per sample)
# y(n) =  Sum(x(n-i))i=1..D * 1/D 
#
# Recursive Running Sum (one addition and one subtraction per sample)
# y(n) = [ x(n) - x(n-D) ] * 1/D + y(n-1)
#
# Cascade Integrator Comb Filter as Moving Average Filter
# y(n) = ( x(n) - x(n-D) ) + y(n-1)
# https://en.wikipedia.org/wiki/Cascaded_integrator-comb_filter
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue
import collections

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
# y(n) = ( x(n) - x(n-D) ) + y(n-1)
@vectorize(['uint8(uint8, uint8, uint8)'], nopython=True, fastmath=True)
def runsum(data, data_delayed, data_previous):
    # x(n), x(n-D) y(n-1)
    return np.add(np.subtract(data, data_delayed), data_previous)

@vectorize(['uint8(uint8, uint8)'], nopython=True, fastmath=True)
def highpass(data_filtered, data):
    return np.subtract(data_filtered, data)

class highpassProcessor(Thread):
    """Highpass filter"""

    # Initialize the Processor Thread
    def __init__(self, res: (int, int, int), delay: int = 1 ):

        # Threading Locks, Events
        self.stopped = True

        # Initialize Processor
        self.data_lowpass  = np.zeros(res, 'float32')
        self.data_highpass = np.zeros(res, 'float32')

        # Init Frame and Thread
        self.input    = Queue(maxsize=32)
        self.output   = Queue(maxsize=32)
        self.log      = Queue(maxsize=32)
        self.stopped  = True

        self.circular_buffer = collections.deque(maxlen=delay)
        # initialize buffer with zeros
        for i in range(delay):
            self.circular_buffer.append(self.data_lowpass)

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
            xn = data                                     # x(n)
            xnd = self.circular_buffer.popleft()          # x(N-D)
            self.circular_buffer.append(data)             # put new data into delay line
            yn1 = self.data_lowpass                       # y(n-1)
            self.data_lowpass = runsum(xn, xnd, yn1)      # y(n) = x(n) - x(n-D) + y(n-1)
            self.data_hihgpass = highpass(self.data_lowpass, data)
            total_time += time.perf_counter() - start_time

            # put results into output queue
            if not self.output.full():
                self.output.put_nowait((data_time, self.data_highpass))
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
                if not self.log.full(): self.log.put_nowait((logging.INFO, "Proc:CPS:{}".format(self.measured_cps)))
                if not self.log.full(): self.log.put_nowait((logging.INFO, "Proc:Time:{}".format(self.measured_time)))
                num_cubes = 0
                total_time = 0
                last_time = current_time
