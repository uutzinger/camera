##########################################################################
# Testing of display and highpass processor
##########################################################################
# Results
##########################################################################

# System
import logging, time, platform
from os import fsdecode
import math

# Matrix Algebra
import numpy as     np
from   numba import vectorize, jit, prange

# OpenCV
import cv2

configs = {
    'camera_res'      : (1280, 720),    # width & height
    'exposure'        : -6,             # -1,0 = auto, 1...max=frame interval, 
    'autoexposure'    : 1.0,            # depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 30,             # 15, 30, 40, 90, 120, 180
    'fourcc'          : -1,             # n.a.
    'buffersize'      : -1,             # n.a.
    'output_res'      : (-1, -1),       # Output resolution, -1,-1 no change
    'flip'            : 0,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'displayfps'       : 30             # frame rate for display server
}

if configs['displayfps'] >= configs['fps']:  display_interval = 0
else:                                        display_interval = 1.0/configs['displayfps']

res = configs['camera_res']
height = res[1]
width = res[0]
camera_index = 0 # default camera starts at 0 by operating system

bin_x = 10
bin_y = 10
scale = (bin_x*bin_y*255)

# transform highpassed data to display image
#
# data = np.sqrt(np.multiply(data,abs(data_bandpass)))
# data = np.sqrt(255.*np.absolute(data_highpass)).astype('uint8')
# data = (128.-data_highpass).astype('uint8')
# data = np.left_shift(np.sqrt((np.multiply(data_lowpass,np.absolute(data_highpass)))).astype('uint8'),2)
@vectorize(['float32(float32)'], nopython=True, fastmath=True)
def displaytrans(data_bandpass):
    return np.sqrt(16.*np.abs(data_bandpass))

@jit(nopython=True, fastmath=True, parallel=True)
def bin4_sum8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//4,n,o), dtype='uint16')
    arr_out = np.empty((m//4,n//4,o), dtype='uint16')
    for i in prange(m//4):
        arr_tmp[i,:,:] =  arr_in[i*4,:,:] +  arr_in[i*4+1,:,:] +  arr_in[i*4+2,:,:] + arr_in[i*4+3,:,:]
    for j in prange(n//4):
        arr_out[:,j,:] = arr_tmp[:,j*4,:] + arr_tmp[:,j*4+1,:] + arr_tmp[:,j*4+2,:] + arr_tmp[:,j*4+3,:]
    return arr_out

@jit(nopython=True, fastmath=True, parallel=True)
def bin10_sum8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//10,n,o), dtype='uint16')
    arr_out = np.empty((m//10,n//10,o), dtype='uint16')
    for i in prange(m//10):
        arr_tmp[i,:,:] =  arr_in[i*10,:,:] +  arr_in[i*10+1,:,:] +  arr_in[i*10+2,:,:] +  arr_in[i*10+3,:,:] +  arr_in[i*10+4,:,:] +  arr_in[i*10+5,:,:] +  arr_in[i*10+6,:,:] +  arr_in[i*10+7,:,:] +  arr_in[i*10+8,:,:] +  arr_in[i*10+9,:,:]
    for j in prange(n//10):
        arr_out[:,j,:] = arr_tmp[:,j*10,:] + arr_tmp[:,j*10+1,:] + arr_tmp[:,j*10+2,:] + arr_tmp[:,j*10+3,:] + arr_tmp[:,j*10+4,:] + arr_tmp[:,j*10+5,:] + arr_tmp[:,j*10+6,:] + arr_tmp[:,j*10+7,:] + arr_tmp[:,j*10+8,:] + arr_tmp[:,j*10+9,:]
    return arr_out

# Processing
data_highpass_h = np.zeros((height, width, 3), 'float32')
data_lowpass_h  = np.zeros((height, width, 3), 'float32')
data_highpass_l = np.zeros((height, width, 3), 'float32')
data_lowpass_l  = np.zeros((height, width, 3), 'float32')

# Setting up logging
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

# Construct poor man's lowpass filter y = (1-alpha) * y + alpha * x
# https://dsp.stackexchange.com/questions/54086/single-pole-iir-low-pass-filter-which-is-the-correct-formula-for-the-decay-coe
# filter cut off frequency at low side 0.5Hz
f_s = configs['fps']                   # sampling frenecy [1/s]
f_c = 0.5/(2.*f_s)                      # normalized cut off frequency
w_c = (2.*3.141)*f_c                   # normalized cut off frequency in radians
y = 1 - math.cos(w_c);                 # compute alpha for 3dB attenuation at cut off frequency
alpha_l = -y + math.sqrt( y*y + 2*y ); # 

# filter cut off frequency at high side 10Hz
f_s = configs['fps']                   # sampling frenecy [1/s]
f_c = 10./(2.*f_s)                      # normalized cut off frequency
w_c = (2.*3.141)*f_c                   # normalized cut off frequency in radians
y = 1 - math.cos(w_c);                 # compute alpha for 3dB attenuation at cut off frequency
alpha_h = -y + math.sqrt( y*y + 2*y ); # 

from camera.processor.highpassprocessor import highpassProcessor
processor_l= highpassProcessor(res=(height//bin_x, width//bin_y, 3), alpha=alpha_l)
processor_l.start()
logger.log(logging.INFO, "Started Processor 05")

processor_h = highpassProcessor(res=(height//bin_x, width//bin_y, 3), alpha=alpha_h)
processor_h.start()
logger.log(logging.INFO, "Started Processor 10")

# Create camera interface
# Computer OS and platform dependent
plat = platform.system()
if plat == 'Linux':
    if platform.machine() == "aarch64": # this is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is raspberry for me
        from camera.capture.cv2capture import cv2Capture
        camera = cv2Capture(configs, camera_index)
else:
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs, camera_index)
camera.start()
logger.log(logging.INFO, "Started Capture")

# Display
main_window_name        = 'Capture'
processed_window_name_1b = 'Band Pass B'
processed_window_name_1g = 'Band Pass G'
processed_window_name_1r = 'High Pass R'
processed_window_name_2 = 'Band Pass'
processed_window_name_3 = 'Low Pass High'
processed_window_name_4 = 'Low Pass Low'
font                    = cv2.FONT_HERSHEY_SIMPLEX
textLocation0           = (10,height-40)
textLocation1           = (10,height-20)
fontScale               = 1
fontColor               = (255,255,255)
lineType                = 2

cv2.namedWindow(main_window_name,         cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(processed_window_name_1b, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(processed_window_name_1g, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(processed_window_name_1r, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(processed_window_name_2,  cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(processed_window_name_3, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(processed_window_name_4, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Initialize Variables
last_display           = time.perf_counter() # keeo track of time to display images
last_time              = time.perf_counter() # keeo track of time to display images
counter                = 0
bin_time  = 0

# Main Loop
stop =  False
while(not stop):
    current_time = time.perf_counter()

    # Camera get data
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)

    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, msg)

    start_time  = time.perf_counter()
    frame_filt  = bin10_sum8(frame)
    bin_time   += (time.perf_counter() - start_time)

    # Processor, put and retrieve data
    if not processor_l.input.full():   processor_l.input.put_nowait((frame_time, frame_filt))
    else:                              logger.log(logging.WARNING, "Proc L:Input Queue is full!")

    if not processor_h.input.full():   processor_h.input.put_nowait((frame_time, frame_filt))
    else:                              logger.log(logging.WARNING, "Proc H:Input Queue is full!")

    if not processor_l.output.empty(): (data_time, data_highpass_l, data_lowpass_l) = processor_l.output.get()

    if not processor_h.output.empty(): (data_time, data_highpass_h, data_lowpass_h) = processor_h.output.get()

    while not processor_l.log.empty():  
        (level, msg) = processor_l.log.get_nowait()
        logger.log(level, msg)

    while not processor_h.log.empty():
        (level, msg) = processor_h.log.get_nowait()
        logger.log(level, msg)

    # Display camera and processed data
    if (current_time - last_display) >= display_interval:

        cv2.putText(frame,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(main_window_name, frame)

        data_lowpass_hs = data_lowpass_h/scale
        data_lowpass_ls = data_lowpass_l/scale
        data_bandpass   = data_lowpass_hs - data_lowpass_ls

        data_bandpass_display   = displaytrans(data_bandpass)        
        data_bandpass_display_r = cv2.resize(data_bandpass_display, (width,height))

        cv2.imshow(processed_window_name_1b, data_bandpass_display[:,:,0])
        cv2.imshow(processed_window_name_1g, data_bandpass_display[:,:,1])
        cv2.imshow(processed_window_name_1r, data_bandpass_display[:,:,2])

        cv2.imshow(processed_window_name_2, data_bandpass_display_r)
        cv2.imshow(processed_window_name_3, data_lowpass_hs)
        cv2.imshow(processed_window_name_4, data_lowpass_ls)

        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True

        last_display = current_time
        counter += 1

    if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
        logger.log(logging.INFO, "Bin:{}".format(bin_time/5.0))
        bin_time = 0
        last_time = current_time

# Cleanup
camera.stop()
processor_l.stop()
processor_h.stop()
cv2.destroyAllWindows()
