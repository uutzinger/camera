##########################################################################
# Testing of python process
##########################################################################
import logging
from multiprocessing import Queue, Process, Event, Lock

# Loop until stop
def loop(queue, stop, lock, logger):
    """ run the process """
    while not stop.is_set():
        # Take data from queue
        data = queue.get(block=True, timeout=None)
        # Do something with data
        # ...
        # Report
        lock.acquire()
        print("Done Something")
        logger.log(logging.INFO, 'Done something') 
        lock.release()

if __name__ == '__main__':
    logger = logging.getLogger("Tester")
    # Threading Locks, Events
    stop = Event()
    lock = Lock()
    stop.clear()

    # Setting up queue
    queue = Queue(maxsize=32)

    print("Setting up Process")
    process=Process(target=loop, args=(queue, stop, lock, logger, ))
    process.daemon = True
    process.start()

    print("Provide data to Process queue")
    data = [0,1,2,3,4,5,6,7]
    if not queue.full():
        queue.put(data, block=False) 

    # Finish
    print("Cleaning up")
    stop.set()
    process.join()
    process.close()
    queue.close()
