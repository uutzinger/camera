###############################################################################
# Highpass Filter
# Urs Utzinger 2022
#
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
from queue import Empty, Queue
import collections
from typing import Optional

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

# Filter implementation
# y(n) = ( x(n) - x(n-D) ) + y(n-1)
@vectorize(['float32(float32, float32, float32)'], nopython=True, fastmath=True)
def runsum(data: np.float32, data_delayed: np.float32, data_previous: np.float32) -> np.float32:
    # x(n), x(n-D), y(n-1)
    return (data - data_delayed) + data_previous


@vectorize(['float32(float32, float32, float32)'], nopython=True, fastmath=True)
def highpass(data: np.float32, data_running_sum: np.float32, inv_delay: np.float32) -> np.float32:
    # highpass = x(n) - lowpass_average
    return data - (data_running_sum * inv_delay)


class runningsumProcessor(Thread):
    """Running-sum lowpass (CIC moving average) + derived highpass."""

    # Initialize the Processor Thread
    def __init__(self, res: tuple[int, int, int], delay: int = 1 ):

        # Threading Locks, Events
        self.stopped = True

        # Initialize Processor
        if delay < 1:
            raise ValueError("delay must be >= 1")
        self.delay = int(delay)
        self.inv_delay = np.float32(1.0 / float(self.delay))

        self.data_lowpass = np.zeros(res, dtype=np.float32)   # running sum
        self.data_highpass = np.zeros(res, dtype=np.float32)  # x - (sum / D)
        self._x = np.zeros(res, dtype=np.float32)

        # Init Frame and Thread
        self.input    = Queue(maxsize=32)
        self.output   = Queue(maxsize=32)
        self.log      = Queue(maxsize=32)
        self.stopped  = True

        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Optional[Thread] = None

        # Delay line holding past x(n) samples as float32 buffers.
        self.circular_buffer = collections.deque(maxlen=self.delay)
        for _ in range(self.delay):
            self.circular_buffer.append(np.zeros(res, dtype=np.float32))

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
            # Convert input to float32 in preallocated buffer.
            np.copyto(self._x, data, casting='unsafe')

            xnd = self.circular_buffer.popleft()  # x(n-D) buffer
            # y(n) = x(n) - x(n-D) + y(n-1)  (running sum)
            runsum(self._x, xnd, self.data_lowpass, out=self.data_lowpass)
            # highpass = x(n) - (y(n)/D)
            highpass(self._x, self.data_lowpass, self.inv_delay, out=self.data_highpass)
            # Reuse popped buffer to store current x(n) and push into delay line.
            np.copyto(xnd, self._x)
            self.circular_buffer.append(xnd)
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
                self.measured_time = (total_time/num_cubes) if num_cubes else 0.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "RunSum:CPS:{}".format(self.measured_cps)))
                if not self.log.full(): self.log.put_nowait((logging.INFO, "RunSum:Time:{}".format(self.measured_time)))
                num_cubes = 0
                total_time = 0
                last_time = current_time


# Backwards compatibility: older code may import `highpassProcessor` from this module.
highpassProcessor = runningsumProcessor
