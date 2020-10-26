import cv2
import logging
import time

# Camera configuration file
from camera.configs.blackfly_configs  import configs
display_interval = 1.0/configs['serverfps']

font                    = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText0 = (10,480)
bottomLeftCornerOfText1 = (10,520)
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
window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE) # or normal
last_display = time.time()
while(cv2.getWindowProperty("Camera", 0) >= 0):
    current_time = time.time()
    if (current_time - last_display) > display_interval:
        if camera.new_frame:
            img = camera.frame
            cv2.putText(img,"Capture FPS:{} [Hz]".format(camera.measured_fps), bottomLeftCornerOfText0, font, fontScale, fontColor, lineType)
            cv2.putText(img,"Display FPS:{} [Hz]".format(configs['serverfps']), bottomLeftCornerOfText1, font, fontScale, fontColor, lineType)
            cv2.imshow('Camera', img)
        last_display = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        pass
camera.stop()
cv2.destroyAllWindows()