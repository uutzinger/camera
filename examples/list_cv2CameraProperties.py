##########################################################################
# List opencv camera properties
##########################################################################
import logging
import cv2

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("CV2")

def decode_fourcc(val):
    """ decode the fourcc integer to the chracter string """
    return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

def cv2SettingsDebug(capture,logger):
    """ return opencv camera properties """
    if cap.isOpened():   
        logger.log(logging.DEBUG, "POS_MSEC:    {}".format(capture.get(cv2.CAP_PROP_POS_MSEC)))
        logger.log(logging.DEBUG, "POS_FRAMES:  {}".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))           # 
        logger.log(logging.DEBUG, "AVI_RATIO:   {}".format(capture.get(cv2.CAP_PROP_POS_AVI_RATIO)))        # 
        logger.log(logging.DEBUG, "WIDTH:       {}".format(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))          # 
        logger.log(logging.DEBUG, "HEIGHT:      {}".format(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))         # 
        logger.log(logging.DEBUG, "FPS:         {}".format(capture.get(cv2.CAP_PROP_FPS)))                  # 
        logger.log(logging.DEBUG, "FOURCC:      {}".format(capture.get(cv2.CAP_PROP_FOURCC)))               # 
        tmp = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))         
        logger.log(logging.DEBUG, "FOURCC:      {}".format(tmp))                                                 # 
        logger.log(logging.DEBUG, "FRAME_CNT:   {}".format(capture.get(cv2.CAP_PROP_FRAME_COUNT)))          # 
        logger.log(logging.DEBUG, "FORMAT:      {}".format(capture.get(cv2.CAP_PROP_FORMAT)))               # 
        logger.log(logging.DEBUG, "MODE:        {}".format(capture.get(cv2.CAP_PROP_MODE)))                 # 
        logger.log(logging.DEBUG, "BRIGHTNESS:  {}".format(capture.get(cv2.CAP_PROP_BRIGHTNESS)))           # 
        logger.log(logging.DEBUG, "CONTRAST:    {}".format(capture.get(cv2.CAP_PROP_CONTRAST)))             #
        logger.log(logging.DEBUG, "SATURATION:  {}".format(capture.get(cv2.CAP_PROP_SATURATION)))           # 
        logger.log(logging.DEBUG, "HUE:         {}".format(capture.get(cv2.CAP_PROP_HUE)))                  # 
        logger.log(logging.DEBUG, "GAIN:        {}".format(capture.get(cv2.CAP_PROP_GAIN)))                 # 
        logger.log(logging.DEBUG, "EXPOSURE:    {}".format(capture.get(cv2.CAP_PROP_EXPOSURE)))             #
        logger.log(logging.DEBUG, "CONV_RGB:    {}".format(capture.get(cv2.CAP_PROP_CONVERT_RGB)))          # 
        logger.log(logging.DEBUG, "RECT:        {}".format(capture.get(cv2.CAP_PROP_RECTIFICATION)))        # 
        logger.log(logging.DEBUG, "MONO:        {}".format(capture.get(cv2.CAP_PROP_MONOCHROME)))           # 
        logger.log(logging.DEBUG, "SHARP:       {}".format(capture.get(cv2.CAP_PROP_SHARPNESS)))            # 
        logger.log(logging.DEBUG, "AUTO_EXP:    {}".format(capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)))        # 
        logger.log(logging.DEBUG, "GAMMA:       {}".format(capture.get(cv2.CAP_PROP_GAMMA)))                # 
        logger.log(logging.DEBUG, "TRIGGER:     {}".format(capture.get(cv2.CAP_PROP_TRIGGER)))              # 
        logger.log(logging.DEBUG, "TRIGGER_DEL: {}".format(capture.get(cv2.CAP_PROP_TRIGGER_DELAY)))        # 
        logger.log(logging.DEBUG, "AUTOWB:      {}".format(capture.get(cv2.CAP_PROP_AUTO_WB)))              # 
        logger.log(logging.DEBUG, "WB_TEMP:     {}".format(capture.get(cv2.CAP_PROP_WB_TEMPERATURE)))       # 
        logger.log(logging.DEBUG, "WB_BLUE:     {}".format(capture.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))) # 
        logger.log(logging.DEBUG, "WB_RED:      {}".format(capture.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)))  # 
        logger.log(logging.DEBUG, "TEMP:        {}".format(capture.get(cv2.CAP_PROP_TEMPERATURE)))          # 
        logger.log(logging.DEBUG, "ZOOM:        {}".format(capture.get(cv2.CAP_PROP_ZOOM)))                 # 
        logger.log(logging.DEBUG, "FOCUS:       {}".format(capture.get(cv2.CAP_PROP_FOCUS)))                # 
        logger.log(logging.DEBUG, "GUID:        {}".format(capture.get(cv2.CAP_PROP_GUID)))                 # 
        logger.log(logging.DEBUG, "ISO:         {}".format(capture.get(cv2.CAP_PROP_ISO_SPEED)))            # 
        logger.log(logging.DEBUG, "BACKLIGHT:   {}".format(capture.get(cv2.CAP_PROP_BACKLIGHT)))            # 
        logger.log(logging.DEBUG, "PAN:         {}".format(capture.get(cv2.CAP_PROP_PAN)))                  # 
        logger.log(logging.DEBUG, "TILT:        {}".format(capture.get(cv2.CAP_PROP_TILT)))                 #
        logger.log(logging.DEBUG, "ROLL:        {}".format(capture.get(cv2.CAP_PROP_ROLL)))                 # 
        logger.log(logging.DEBUG, "IRIS:        {}".format(capture.get(cv2.CAP_PROP_IRIS)))                 # 
        logger.log(logging.DEBUG, "SETTINGS:    {}".format(capture.get(cv2.CAP_PROP_SETTINGS)))             # 
        logger.log(logging.DEBUG, "BUFFERSIZE:  {}".format(capture.get(cv2.CAP_PROP_BUFFERSIZE)))           # 
        logger.log(logging.DEBUG, "AUTOFOCUS:   {}".format(capture.get(cv2.CAP_PROP_AUTOFOCUS)))            # 
        logger.log(logging.DEBUG, "SAR_NUM:     {}".format(capture.get(cv2.CAP_PROP_SAR_NUM)))              # 
        logger.log(logging.DEBUG, "SAR_DEN:     {}".format(capture.get(cv2.CAP_PROP_SAR_DEN)))              # 
        logger.log(logging.DEBUG, "BACKEND:     {}".format(capture.get(cv2.CAP_PROP_BACKEND)))              # 
        logger.log(logging.DEBUG, "CHANNEL:     {}".format(capture.get(cv2.CAP_PROP_CHANNEL)))              # 
    else: 
        logger.log(logging.DEBUG, "NaN")

# check for up to 10 cameras
index = 0
arr = []
i = 10
while i > 0:
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        logger.log(logging.DEBUG, "Camera {}:".format(index))
        logger.log(logging.DEBUG, "=============================")
        cv2SettingsDebug(cap,logger)
    index += 1
    i -= 1
