import h5py
import cv2
import logging
import time
import numpy as np
from datetime import datetime

# Camera configuration file
from camera.configs.blackfly_configs  import configs
display_interval = 0.5

font                    = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText0 = (10,480)
bottomLeftCornerOfText1 = (10,520)
fontScale               = 1
fontColor               = (255,255,255)
lineType                = 2
    
# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING

# Setting up Storage
from camera.streamer.storageserver import h5Server
print("Starting Storage Server")
now = datetime.now()
filename = now.strftime("%Y%m%d%H%M%S") + ".hdf5"
hdf5 = h5Server("C:\\temp\\" + filename)
print("Starting Storage Server")
hdf5.start()
# data_cube = np.zeros((540,720,14), dtype=np.uint8)
data_cube = np.random.randint(0, 255, (540, 720, 14), 'uint8')

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
camera.start()

# Initialize Variables
last_display = time.time()
frame_idx = 0     # 14 frames is one dataset
looptime = 0.0005

# Main Loop
while(True):
    current_time = time.time()
    # if new image, store it in data cube 
    if camera.new_frame:
        #tic_0 = time.time()
        data_cube[:,:,frame_idx] = camera.frame # 1-3 ms
        frame_idx += 1
        #tic_1 = time.time()
    # we have a complete dataset, store it and reset frame index , takes 0-1ms
    if frame_idx >= 14: 
        frame_idx = 0  
        hdf5.framearrayTime = camera.frameTime # data array will be saved with this name
        hdf5.framearray = data_cube # transfer array over to storage server
    # delay_time = looptime - (current_time - time.time()) # execute this loop no more than XX times per second
    # if  delay_time > 0.0:
    #    time.sleep(delay_time)
# Cleanup
camera.stop()
hdf5.stop()
cv2.destroyAllWindows()