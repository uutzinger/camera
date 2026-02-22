###############################################################################
# Highpass Filter
# Urs Utzinger 2022
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Empty, Queue
from typing import Optional

#
import logging
import time
#
import numpy as np
from numba import vectorize
from scipy.signal import lfilter
class OnePoleHighpass:
    # alpha = exp(-2πfc/fs)
    def __init__(self, alpha):
        # b = [1, -1], a = [1, -alpha]
        self.b = [1.0, -1.0]
        self.a = [1.0, -alpha]
        self.zi = None

    def highpass(self, x):
        y, self.zi = lfilter(self.b, self.a, x, axis=-1, zi=self.zi)
        return y

###############################################################################
# High Pass Image Processor
# Poor Man's
###############################################################################
# Construct poor man's low pass filter y = (1-alpha) * y + alpha * x
# https://dsp.stackexchange.com/questions/54086/single-pole-iir-low-pass-filter-which-is-the-correct-formula-for-the-decay-coe
# f_s sampling frequency in Hz
# f_c cut off frequency in Hz
# w_c radians 0..pi (pi is Niquist fs/2, 2pi is fs)
###############################################################################
# Urs Utzinger 2022


###############################################################################
# Basic Image Processor
###############################################################################

# Construct poor man's low pass filter 
# y = (1-alpha) * y + alpha * x
#
# https://dsp.stackexchange.com/questions/54086/single-pole-iir-low-pass-filter-which-is-the-correct-formula-for-the-decay-coe
# f_s sampling frequency in Hz
# f_c cut off frequency in Hz
# w_c radians 0..pi (pi is Niquist fs/2, 2pi is fs)

# Numpy Vectorized Image Processor
# y = (1-alpha) * y + alpha * x
@vectorize(['float32(uint16, float32, float32)'], nopython=True, fastmath=True)
def movingavg(data, average, alpha):
    return np.add(np.multiply(average, 1.-alpha), np.multiply(data, alpha))

# Highpass Filter
@vectorize(['float32(uint16, float32)'], nopython=True, fastmath=True)
def highpass(data, average):
    return np.subtract(data, average)

def computeAlpha(f_s: float = 50.0, f_c: float = 5.0) -> float:
    """Alpha for y=(1-alpha)*y_prev + alpha*x with -3 dB at f_c."""
    if f_s <= 0.0:
        raise ValueError("f_s must be > 0")
    if f_c <= 0.0 or f_c >= (0.5 * f_s):
        raise ValueError("f_c must satisfy 0 < f_c < f_s/2")
    from math import cos, pi, sqrt
    w_c = (2.0 * pi) * f_c / f_s
    t = 1.0 - cos(w_c)
    return float(-t + sqrt((t * t) + (2.0 * t)))
    
class highpassProcessor(Thread):
    """Highpass filter"""

    # Initialize the Processor Thread
    def __init__(self, res: tuple[int, int, int], alpha: float = 0.95 ):

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

        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Optional[Thread] = None

        Thread.__init__(self)

    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def start(self):
        """set the thread start conditions"""
        if not self.stopped:
            return

        self.stopped = False
        self._thread = Thread(target=self.update)
        self._thread.daemon = True # run in background
        self._thread.start()

    # After Starting the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_cubes = 0
        total_time = 0.0
        while not self.stopped:

            ############################################################################
            # Image Processing
            ############################################################################
            # obtain new data
            try:
                (data_time, data) = self.input.get(block=True, timeout=0.25)
            except Empty:
                continue
            # process
            start_time = time.perf_counter()
            movingavg(data, self.averageData, self.alpha, out=self.averageData)
            highpass(data, self.averageData, out=self.filteredData)
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
                self.measured_time = (total_time/num_cubes) if num_cubes else 0.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "HighPass:CPS:{}".format(self.measured_cps)))
                if not self.log.full(): self.log.put_nowait((logging.INFO, "HighPass:Time:{}".format(self.measured_time)))
                num_cubes = 0
                total_time = 0
                last_time = current_time
