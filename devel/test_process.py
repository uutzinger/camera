##########################################################################
# Testing of python process
# Copies 2500 x 3 x 720 x 540 bytes/sec = 2.7Gbytes/sec
# Python 1: 10%, Python 2 9.5% (8 Cores 16 Threads)
##########################################################################
import logging, time
from multiprocessing import Queue, Process, Event, Lock
import cv2
import numpy as np

class Tester(Process):
    """Process Class"""
    # Initialize
    def __init__(self,queue,log):
        Process.__init__(self)
        self.stopper = Event()
        self.locker = Lock()
        self.queue = queue
        self.log = log

    # Stop
    def stop(self):
        """Stop the process"""
        self.stopper.set()
        self.process.join()
        self.process.close()

    # Start
    def start(self):
        """Setup the process and start it"""
        self.stopper.clear()
        self.process=Process(target=self.loop)
        self.process.daemon = True
        self.process.start()

    # Loop until stop
    def loop(self):
        """Run the process"""
        # cv2.namedWindow('Data', cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
        font          = cv2.FONT_HERSHEY_SIMPLEX
        textLocation0 = (10,20)
        fontScale     = 1
        fontColor     = (255,255,255)
        lineType      = 2
        last_time = time.time()
        num_frames = 0

        while not self.stopper.is_set():
            current_time = time.time()
            # Take data from queue
            try:
                img = self.queue.get(block=True, timeout=0.1)
                num_frames += 1
                #cv2.putText(img,"Frame:{}".format(num_frames),             textLocation0, font, fontScale, fontColor, lineType)
                #cv2.imshow('Data', img)
                #if cv2.waitKey(1) & 0xFF == ord('q'): 
                #    self.stopper.set()
                #    break
            except:
                pass

            # update displayed frames per second
            if current_time - last_time >= 5.:
                dps = num_frames/(current_time-last_time)
                last_time = current_time
                num_frames = 0
                self.log.put_nowait((logging.INFO,"Status:Frames received per second:{}".format(dps)))

if __name__ == '__main__':

    width = 720       # 1920, 720
    height = 540      # 1080, 540

    # synthetic data
    img = np.random.randint(0, 255, (height, width), 'uint8') # random image

    logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
    logger = logging.getLogger('Tester')

    print("Setting up process")
    queue = Queue(maxsize=32)
    log = Queue()
    tester = Tester(queue, log)
    tester.start()

    while not tester.stopper.is_set():
        try: queue.put_nowait(img) 
        except: pass

        try: 
            (level, msg) = log.get_nowait()
            logger.log(level, "Status:{}".format(msg))
        except: pass

    # Finish
    print("Cleaning up")
    tester.stop()
    queue.close()
    log.close()
