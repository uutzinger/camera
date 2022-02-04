###############################################################################
# Storage array data streamer
# Urs Utzinger 2020
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue

# System
import logging, time

# TIFF
import tifffile


###############################################################################
# TIFF Storage Server
###############################################################################

class tiffServer(Thread):
    """
    Tiff saver
    """

    # Initialize the storage Thread
    # Opens Capture Device
    def __init__(self, filename = None):

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # Initialize TIFF
        if filename is not None:
            try:
                self.tiff = tifffile.TiffWriter(filename, bigtiff=True)
            except:
                if not self.log.full(): self.log.put_nowait((logging.ERROR, "TIFF:Could not create TIFF!"))
                return False
        else:
            if not self.log.full(): self.log.put_nowait((logging.ERROR, "TIFF:Need to provide filename to store data!"))
            return False

        self.measured_cps = 0.0

        Thread.__init__(self)

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update)
        T.daemon = True # run in background
        T.start()

    # After Stating of the Thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_cubes = 0

        while not self.stopped:
            (cube_time, data_cube) = self.queue.get(block=True, timeout=None)
            self.tiff.write(data_cube, compression='PACKBITS', photometric='MINISBLACK', contiguous=False, metadata ={'time': cube_time, 'author': 'camera'} )
            # compression = 'LZW', 'LZMA', 'ZSTD', 'JPEG', 'PACKBITS', 'NONE', 'LERC'
            # compression ='jpeg', 'png', 'zlib'
            num_cubes += 1


            # Storage througput calculation
            current_time = time.time()
            if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
                self.measured_cps = num_cubes/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "TIFF:CPS:{}".format(self.measured_cps)))
                num_cubes = 0
                last_time = current_time

        self.tiff.close()

if __name__ == '__main__':

    import numpy as np
    from datetime import datetime
    import cv2

    display_interval =  0.01
    height =540
    width = 720
    depth = 3 # can display only 3 colors ;-)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("TIFF")
   
    # Setting up Storage
    now = datetime.now()
    filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
    tiff = tiffServer("C:\\temp\\" + filename)
    logger.log(logging.DEBUG, "Starting TIFF Server")
    tiff.start()

    # synthetic image
    image = np.random.randint(0, 255, (depth, height, width), 'uint8') 

    window_handle = cv2.namedWindow("TIFF", cv2.WINDOW_AUTOSIZE)
    font          = cv2.FONT_HERSHEY_SIMPLEX
    textLocation  = (10,20)
    fontScale     = 1
    fontColor     = (255,255,255)
    lineType      = 2

    last_display = time.time()
    num_frames = 0
    while(cv2.getWindowProperty("TIFF", 0) >= 0):
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            last_display = current_time
            frame = image[1,:,:].copy()
            cv2.putText(frame,"Frame:{}".format(num_frames), textLocation, font, fontScale, fontColor, lineType)
            cv2.imshow('TIFF', frame)
            num_frames += 1
            key = cv2.waitKey(1) 
            if (key == 27) or (key & 0xFF == ord('q')): break

            try: tiff.queue.put_nowait((current_time, image))
            except: pass

        while not tiff.log.empty():
            (level, msg)=tiff.log.get_nowait()
            logger.log(level, "TIFF:{}".format(msg))

    tiff.stop()
    cv2.destroyAllWindows()
