##########################################################################
# Testing of python process
##########################################################################
import logging
from multiprocessing import Queue, Process, Event, Lock

class Tester(Process):
    """Process Class"""
    # Initialize
    def __init__(self):
        Process.__init__(self)
        # Initialize logger
        self.logger = logging.getLogger("Tester")
        # Threading Locks, Events
        self.stop = Event()
        self.lock = Lock()
        self.stop.clear()

    # Loop until stop
    def loop(self, queue):
        """ run the process """
        while not self.stop.is_set():
            # Take data from queue
            data = queue.get(block=True, timeout=None)
            # Do something with data
            # ...
            # Report
            self.lock.aquire()
            self.logger.log(logging.INFO, 'Done something') 
            self.lock.release()

# Setting up queue
my_queue = Queue(maxsize=32)

print("Setting up Process")
my_tester = Tester(my_queue)
process=Process(target=my_tester.loop, args=(my_queue,) )
# self.process.daemon = True
process.start()

print("Provide data to Process queue")
data = [0,1,2,3,4,5,6,7]
if not my_queue.full():
    my_queue.put(data, block=False) 

# Finish
print("Cleaning up")
my_tester.stop.set()
my_tester.close()
my_queue.close()
