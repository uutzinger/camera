###############################################################################
# Background, Flatfield and White Balance Processor
# Urs Utzinger 2022
###############################################################################

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

# Multi Threading
from threading import Thread
from queue import Empty, Queue

import logging
import time
#
import numpy as np
from numba import vectorize

###############################################################################
# Basic Image Processor
###############################################################################
# Numpy Vectorized Image Processor
@vectorize(['uint16(uint8, uint8, uint16)'], nopython=True, fastmath=True)
def bgflat(pixel: np.uint8, background: np.uint8, flatfield: np.uint16) -> np.uint16:
    value = int(pixel) - int(background)
    if value < 0:
        value = 0
    out = value * int(flatfield)
    if out > 65535:
        out = 65535
    return out

class bgflatProcessor(Thread):
    """Background removal, flat field correction, white balance """

    # Initialize the Processor Thread
    def __init__(
        self,
        flatfield: np.ndarray,
        res: tuple[int, int, int] = (14, 540, 720),
        bg_delta: tuple[int, int] = (64, 64),
    ):
        self.logger = logging.getLogger("bgflatProcessor")

        depth, height, width = res

        if flatfield is None:
            raise ValueError("Need to provide flatfield")

        flatfield_arr = np.asarray(flatfield)
        if flatfield_arr.ndim != 2 or flatfield_arr.shape != (height, width):
            raise ValueError(f"flatfield must have shape ({height}, {width}), got {flatfield_arr.shape}")

        self.ff = flatfield_arr.astype(np.uint16, copy=False)

        # Initialize Processor
        self.bg_dx = int(bg_delta[1])
        self.bg_dy = int(bg_delta[0])

        # Avoid overflow when summing sampled pixels
        self.inten = np.zeros(depth, dtype=np.uint32)
        self.bg = np.zeros((height, width), dtype=np.uint8)
        self.data_cube_corr = np.zeros((depth, height, width), dtype=np.uint16)

        # Queues (optional use; callers may also pass their own queues to start())
        self.input = Queue(maxsize=32)
        self.output = Queue(maxsize=32)
        self.log = Queue(maxsize=64)

        # Thread state
        self.stopped = True
        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Thread | None = None
        self._input_queue = self.input
        self._output_queue = self.output

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def start(self, input_queue: Queue | None = None, output_queue: Queue | None = None):
        """Set the thread start conditions.

        Backwards compatible: callers may provide external queues.
        If omitted, uses self.input/self.output.
        """
        if not self.stopped:
            return

        self._input_queue = input_queue if input_queue is not None else self.input
        self._output_queue = output_queue if output_queue is not None else self.output

        self.stopped = False
        self._thread = Thread(target=self.update)
        self._thread.daemon = True  # run in background
        self._thread.start()

    # After Starting the Thread, this runs continuously
    def update(self):
        """ run the thread """
        input_queue = self._input_queue
        output_queue = self._output_queue

        last_cps_time = time.time()
        num_cubes = 0
        total_time = 0.0
        while not self.stopped:
            # Processing throughput calculation
            current_time = time.time()
            if (current_time - last_cps_time) >= 5.0: # frame array rate every 5 secs
                self.measured_cps = num_cubes/5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"BgFlat:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"BgFlat:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_cps_time = current_time

            try:
                (cube_time, data_cube) = input_queue.get(block=True, timeout=0.25)
            except Empty:
                continue

            # Find background image (minimum sampled intensity across frames)
            sampled = data_cube[:, :: self.bg_dy, :: self.bg_dx]
            self.inten[:] = np.sum(sampled, axis=(1, 2), dtype=np.uint32)
            background_indx = int(np.argmin(self.inten))
            self.bg[:, :] = data_cube[background_indx, :, :]

            # Process
            start_time = time.perf_counter()
            bgflat(data_cube, self.bg, self.ff, out=self.data_cube_corr)
            total_time += time.perf_counter() - start_time

            # Put results into output queue
            if not output_queue.full():
                output_queue.put_nowait((cube_time, self.data_cube_corr))
            else:
                if not self.log.full():
                    self.log.put_nowait((logging.WARNING, "BgFlat:Processed Output Queue is full!"))
            num_cubes += 1
 
