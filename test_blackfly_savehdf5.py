import cv2
import logging
import time
import h5py

# Camera configuration file
from camera.configs.blackfly_configs  import configs
display_interval = 1.0/configs['serverfps']

font                    = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText  = (10,500)
fontScale               = 1
fontColor               = (255,255,255)
lineType                = 2
    
# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING

# Create camera interface
from camera.capture.blackflycapture import blackflyCapture
print("Starting Capture")
camera = blackflyCapture(configs)
print("Getting Images")
camera.start()

f = h5py.File('C:\temp\imager.hdf5','w')
grp = f.create_group("session1")

window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE) # or normal
last_display = time.time()
last_save = last_display
while(cv2.getWindowProperty("Camera", 0) >= 0):
    current_time = time.time()
    if camera.new_frame:
        dset = f.create_dataset(str(camera.frametime), data=camera.frame, compression="lzf")
        num_frames +=1
    if (current_time - last_saved) >= 5.0:
        print("Saving FPS:{} Hz".format(num_frames/5.0)) 
        last_saved = current_time
        num_frames = 0
    if (current_time - last_display) > display_interval:
            img = camera.frame
            cv2.putText(img,"Capture FPS:{} [Hz]".format(camera.measured_fps), bottomLeftCornerOfText0, font, fontScale, fontColor, lineType)
            cv2.imshow('Camera', img)
            last_display = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        pass
camera.stop()
cv2.destroyAllWindows()