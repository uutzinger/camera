###############################################################################
# Background, Flatfield and White Balance Processor
# Urs Utzinger 2022
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock

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
@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def bgflat(data_cube, background, whitefield):
    return np.multiply(np.subtract(data_cube, background), whitefield)

class bgflatProcessor(Thread):
    """Background removal, flat field correction, white balance """

    # Initialize the Processor Thread
    def __init__(self, whitefield, res: (int, int, int) = (14, 540, 720), bg_delta: (int, int) = (64, 64) ):
        # initialize logger 
        self.logger = logging.getLogger("bgflatProcessor")

        # Threading Locks, Events
        self.stopped = True
        self.framearray_lock = Lock()

        # Initialize Processor
        self.bg_dx = bg_delta[1]
        self.bg_dy = bg_delta[0]
        self.inten = np.zeros(res[0], 'uint16') 
        self.bg    = np.zeros((res[1],res[2]), 'uint8')
        self.data_cube_corr = np.zeros(res, 'uint16') 
        if whitefield is None:
            self.logger.log(logging.ERROR, "Status:Need to provide flatfield!")
            return False
        else: self.wf = whitefield

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
                (cube_time, data_cube) = input_queue.get(block=True, timeout=None)
                # Find background image
                _ = np.sum(data_cube[:,::self.bg_dx,::self.bg_dy], axis=(1,2), out = self.inten)
                background_indx = np.argmin(self.inten) # search for minimum intensity 
                self.bg = data_cube[background_indx, :, :]
                # process
                bgflat(data_cube, self.wf, self.bg, out = self.data_cube_corr)
                # put results into output queue
                if not output_queue.full():
                    output_queue.put((cube_time, self.data_cube_corr), block=False)
                else:
                    self.logger.log(logging.WARNING, "Status:Processed Output Queue is full!")
                num_cubes += 1
 