import h5py
import logging
import time
import numpy as np
from datetime import datetime
    
# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING

# Setting up Storage
from camera.streamer.storageserver import h5Server
print("Settingup Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("C:\\temp\\" + filename)
print("Starting Storage Server")
hdf5.start()

data_cube = np.zeros((540,720,14), dtype=np.uint8)

looptime = 0.012

# Main Loop
while True:
    current_time = time.time()
    hdf5.framearrayTime = current_time # data array will be saved with this name
    hdf5.framearray = data_cube # transfer array over to storage server
    delay_time = looptime - (current_time - time.time()) # execute this loop no more than XX times per second
    if  delay_time > 0.0:
        time.sleep(delay_time)
# Cleanup
hdf5.stop()
cv2.destroyAllWindows()