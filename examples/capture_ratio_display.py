##########################################################################
# Testing of display and highpass processor
##########################################################################
# Results
##########################################################################

# System
import logging, time, platform, os
import math

# Matrix Algebra
import numpy as     np
from   numba import vectorize, jit, prange

# OpenCV
import cv2

# Setting up logging
logging.basicConfig(level=logging.INFO) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Main")

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
width  = res[0]
depth  = 3
camera_index = 0 # default camera starts at 0 by operating system

# Reducing the image resolution by binning (summing up pixels)
bin_x=20
bin_y=20
scale = (bin_x*bin_y*255)

# General purpose binning, this is 3 times slower compared to the routines below
def rebin(arr, bin_x=10, bin_y=10, dtype=np.uint16):
    # this only works if array shape is multiple of bin_x and bin_y
    shape = (arr.shape[0] // bin_x, bin_x, arr.shape[1] // bin_y, bin_y)    
    arr_ = arr.astype(dtype)
    return arr_.reshape(shape).sum(-1).sum(1) # sum over last axis and first axis

# Binning 2 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin2(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//2,n,o), dtype='uint16')
    arr_out = np.empty((m//2,n//2,o), dtype='uint16')
    for i in prange(m//2):
        arr_tmp[i,:,:] =  arr_in[i*2,:,:] +  arr_in[i*2+1,:,:]
    for j in prange(n//2):
        arr_out[:,j,:] = arr_tmp[:,j*2,:] + arr_tmp[:,j*2+1,:] 
    return arr_out

# Binning 4 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin4(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//4,n,o), dtype='uint16')
    arr_out = np.empty((m//4,n//4,o), dtype='uint16')
    for i in prange(m//4):
        arr_tmp[i,:,:] =  arr_in[i*4,:,:] +  arr_in[i*4+1,:,:] +  arr_in[i*4+2,:,:] +  arr_in[i*4+3,:,:]
    for j in prange(n//4):
        arr_out[:,j,:] = arr_tmp[:,j*4,:] + arr_tmp[:,j*4+1,:] + arr_tmp[:,j*4+2,:] + arr_tmp[:,j*4+3,:]
    return arr_out

# Binning 8 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//8,n,o), dtype='uint16')
    arr_out = np.empty((m//8,n//8,o), dtype='uint16')
    for i in prange(m//8):
        arr_tmp[i,:,:] =  arr_in[i*8,:,:] +  arr_in[i*8+1,:,:] +  arr_in[i*8+2,:,:] +  arr_in[i*8+3,:,:] +  arr_in[i*8+4,:,:] +  arr_in[i*8+5,:,:] + \
                          arr_in[i*8+6,:,:] +  arr_in[i*8+7,:,:] 
    for j in prange(n//8):
        arr_out[:,j,:] = arr_tmp[:,j*8,:] + arr_tmp[:,j*8+1,:] + arr_tmp[:,j*8+2,:] + arr_tmp[:,j*8+3,:] + arr_tmp[:,j*8+4,:] + arr_tmp[:,j*8+5,:] + \
                         arr_tmp[:,j*8+6,:] + arr_tmp[:,j*8+7,:] 
    return arr_out

# Binning 10 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin10(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//10,n,o), dtype='uint16')
    arr_out = np.empty((m//10,n//10,o), dtype='uint16')
    for i in prange(m//10):
        arr_tmp[i,:,:] =  arr_in[i*10,:,:] +  arr_in[i*10+1,:,:] +  arr_in[i*10+2,:,:] +  arr_in[i*10+3,:,:] +  arr_in[i*10+4,:,:] +  arr_in[i*10+5,:,:] + \
                          arr_in[i*10+6,:,:] +  arr_in[i*10+7,:,:] +  arr_in[i*10+8,:,:] +  arr_in[i*10+9,:,:]
    for j in prange(n//10):
        arr_out[:,j,:] = arr_tmp[:,j*10,:] + arr_tmp[:,j*10+1,:] + arr_tmp[:,j*10+2,:] + arr_tmp[:,j*10+3,:] + arr_tmp[:,j*10+4,:] + arr_tmp[:,j*10+5,:] + \
                         arr_tmp[:,j*10+6,:] + arr_tmp[:,j*10+7,:] + arr_tmp[:,j*10+8,:] + arr_tmp[:,j*10+9,:]
    return arr_out

# Binning 15 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin15(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//15,n,o), dtype='uint16')
    arr_out = np.empty((m//15,n//15,o), dtype='uint32')
    for i in prange(m//15):
        arr_tmp[i,:,:] =  arr_in[i*15,:,:]    + arr_in[i*15+1,:,:]  + arr_in[i*15+2,:,:]  + arr_in[i*15+3,:,:]  + arr_in[i*15+4,:,:]  + arr_in[i*15+5,:,:]  + \
                          arr_in[i*15+6,:,:]  + arr_in[i*15+7,:,:]  + arr_in[i*15+8,:,:]  + arr_in[i*15+9,:,:]  + arr_in[i*15+10,:,:] + arr_in[i*15+11,:,:] + \
                          arr_in[i*15+12,:,:] + arr_in[i*15+13,:,:] + arr_in[i*15+14,:,:] 

    for j in prange(n//15):
        arr_out[:,j,:]  = arr_tmp[:,j*15,:]    + arr_tmp[:,j*15+1,:]  + arr_tmp[:,j*15+2,:]  + arr_tmp[:,j*15+3,:]  + arr_tmp[:,j*10+4,:]  + arr_tmp[:,j*15+5,:]  + \
                          arr_tmp[:,j*15+6,:]  + arr_tmp[:,j*15+7,:]  + arr_tmp[:,j*15+8,:]  + arr_tmp[:,j*15+9,:]  + arr_tmp[:,j*15+10,:] + arr_tmp[:,j*15+11,:] + \
                          arr_tmp[:,j*15+12,:] + arr_tmp[:,j*15+13,:] + arr_tmp[:,j*10+14,:]
    return arr_out

# Binning 20 pixels of the 8bit images
@jit(nopython=True, fastmath=True, parallel=True)
def bin20(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//20,n,o), dtype='uint16')
    arr_out = np.empty((m//20,n//20,o), dtype='uint32')
    for i in prange(m//20):
        arr_tmp[i,:,:] =  arr_in[i*20,:,:]    + arr_in[i*20+1,:,:]  + arr_in[i*20+2,:,:]  + arr_in[i*20+3,:,:]  + arr_in[i*20+4,:,:]  + arr_in[i*20+5,:,:]  + \
                          arr_in[i*20+6,:,:]  + arr_in[i*20+7,:,:]  + arr_in[i*20+8,:,:]  + arr_in[i*20+9,:,:]  + arr_in[i*20+10,:,:] + arr_in[i*20+11,:,:] + \
                          arr_in[i*20+12,:,:] + arr_in[i*20+13,:,:] + arr_in[i*20+14,:,:] + arr_in[i*20+15,:,:] + arr_in[i*20+16,:,:] + arr_in[i*20+17,:,:] + \
                          arr_in[i*20+18,:,:] + arr_in[i*20+19,:,:]

    for j in prange(n//20):
        arr_out[:,j,:]  = arr_tmp[:,j*20,:]    + arr_tmp[:,j*20+1,:]  + arr_tmp[:,j*20+2,:]  + arr_tmp[:,j*20+3,:]  + arr_tmp[:,j*10+4,:]  + arr_tmp[:,j*20+5,:]  + \
                          arr_tmp[:,j*20+6,:]  + arr_tmp[:,j*20+7,:]  + arr_tmp[:,j*20+8,:]  + arr_tmp[:,j*20+9,:]  + arr_tmp[:,j*20+10,:] + arr_tmp[:,j*20+11,:] + \
                          arr_tmp[:,j*20+12,:] + arr_tmp[:,j*20+13,:] + arr_tmp[:,j*10+14,:] + arr_tmp[:,j*20+15,:] + arr_tmp[:,j*20+16,:] + arr_tmp[:,j*20+17,:] + \
                          arr_tmp[:,j*20+18,:] + arr_tmp[:,j*20+19,:] 
    return arr_out


# Transform band passed data to display image
# Goal is to enhace small changes and to convert data to 0..1 range
# A few example options:
# data = np.sqrt(np.multiply(data,abs(data_bandpass)))
# data = np.sqrt(255.*np.absolute(data_highpass)).astype('uint8')
# data = (128.-data_highpass).astype('uint8')
# data = np.left_shift(np.sqrt((np.multiply(data_lowpass,np.absolute(data_highpass)))).astype('uint8'),2)
@vectorize(['float32(float32)'], nopython=True, fastmath=True)
def displaytrans(data_bandpass):
    return np.sqrt(16.*np.abs(data_bandpass))

# Reserve space for process data
data_highpass_h = np.zeros((height//bin_y, width//bin_x, depth), 'float32')
data_lowpass_h  = np.zeros((height//bin_y, width//bin_x, depth), 'float32')
data_highpass_l = np.zeros((height//bin_y, width//bin_x, depth), 'float32')
data_lowpass_l  = np.zeros((height//bin_y, width//bin_x, depth), 'float32')

# Construct poor man's lowpass filter y = (1-alpha) * y + alpha * x
# https://dsp.stackexchange.com/questions/54086/single-pole-iir-low-pass-filter-which-is-the-correct-formula-for-the-decay-coe

# Filter LOW bound
# filter cut off frequency: 0.5Hz
edge = 0.5
f_s = configs['fps']                   # sampling frenecy [1/s]
f_c = edge/(2.*f_s)                    # normalized cut off frequency
w_c = (2.*3.141)*f_c                   # normalized cut off frequency in radians
y = 1 - math.cos(w_c);                 # compute alpha for 3dB attenuation at cut off frequency
alpha_l = -y + math.sqrt( y*y + 2*y ); # 

# Filter HIGH bound
# filter cut off frequency: 10Hz
edge =10.
f_s = configs['fps']                   # sampling frenecy [1/s]
f_c = edge/(2.*f_s)                    # normalized cut off frequency
w_c = (2.*3.141)*f_c                   # normalized cut off frequency in radians
y = 1 - math.cos(w_c);                 # compute alpha for 3dB attenuation at cut off frequency
alpha_h = -y + math.sqrt( y*y + 2*y ); # 

# # Processing Threads
# # LOW Processor
# from camera.processor.highpassprocessor import highpassProcessor
# processor_l= highpassProcessor(res=(height//bin_x, width//bin_y, depth), alpha=alpha_l)
# processor_l.start()
# logger.log(logging.INFO, "Started Processor Lower Bound")
# # HIGH processor
# processor_h = highpassProcessor(res=(height//bin_x, width//bin_y, depth), alpha=alpha_h)
# processor_h.start()
# logger.log(logging.INFO, "Started Processor Upper Bound")


# Create camera interface
# Computer OS and platform dependent
plat = platform.system()

if plat == 'Linux':
    sysname, nodename, release, version, machine = os.uname()
    release == release.split('.')
    if platform.machine() == "aarch64": # this is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    elif platform.machine() == "armv6l" or platform.machine() == 'armv7l': # this is raspberry for me
        if release[0] == 5:
            from camera.capture.libcamcapture import libcameraCapture
            camera = libcameraCapture(configs)            
        else:
            from camera.capture.cv2capture import cv2Capture
            camera = cv2Capture(configs, camera_index)
else:
    from camera.capture.cv2capture import cv2Capture
    camera = cv2Capture(configs, camera_index)
        
camera.start()
logger.log(logging.INFO, "Started Capture")

# Display
main_window_name         = 'Captured'
binned_window_name       = 'Binned'
# processed_window_name    = 'Band-Passed'
ratioed_window_name      = 'Ratioed'

font                     = cv2.FONT_HERSHEY_SIMPLEX
textLocation0            = (10,height-40)
textLocation1            = (10,height-20)
fontScale                = 1
fontColor                = (255,255,255)
lineType                 = 2

cv2.namedWindow(main_window_name,      cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(binned_window_name,    cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
# cv2.namedWindow(processed_window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL
cv2.namedWindow(ratioed_window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Initialize Variables
last_display = last_time = time.perf_counter() # keep track of time to display images
counter      = bin_time  = 0 
stop                     = False 

while(not stop):
    current_time = time.perf_counter()

    # Camera get data
    #################################
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    # Take care of camera log messages
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, msg)

    # Bin the image
    #################################
    start_time  = time.perf_counter()
    frame_bin   = bin20(frame)
    bin_time   += (time.perf_counter() - start_time)

    # Compute Ratio Image 0 = blue, 1 = green, 2 = red
    #################################
    # make the result uint16 because lowpass filter needs uint16 input
    # green/1 over red/2 
    frame_ratio = (frame_bin[:,:,1].astype(np.float32)/frame_bin[:,:,2].astype(np.float32)*255.0).astype(np.uint16)
        
    # # Band Pass the Image
    # # Does not work yet
    # #################################
    # # send new frame to 0.5Hz filter
    # if not processor_l.input.full():   processor_l.input.put_nowait((frame_time, frame_ratio))
    # else:                              logger.log(logging.WARNING, "Proc L:Input Queue is full!")
    # # obtain filtered image
    # if not processor_l.output.empty(): (data_time, data_highpass_l, data_lowpass_l) = processor_l.output.get()
    # # send new frame to 10Hz filter
    # if not processor_h.input.full():   processor_h.input.put_nowait((frame_time, frame_ratio))
    # else:                              logger.log(logging.WARNING, "Proc H:Input Queue is full!")
    # # obtain filtered image
    # if not processor_h.output.empty(): (data_time, data_highpass_h, data_lowpass_h) = processor_h.output.get()
    # # handle processor log messages
    # while not processor_l.log.empty():  
    #     (level, msg) = processor_l.log.get_nowait()
    #     logger.log(level, msg)
    # while not processor_h.log.empty():
    #     (level, msg) = processor_h.log.get_nowait()
    #     logger.log(level, msg)
    
    # Display camera and processed data
    #################################
    if (current_time - last_display) >= display_interval:

        # Display Camera Data
        cv2.putText(frame,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(main_window_name, frame)

        # Display Binned Image, make it same size as original image
        frame_bin = frame_bin/scale # make image 0..1
        frame_tmp = cv2.resize(frame_bin, (width,height), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(binned_window_name, frame_tmp)

        # Display Ratio Image, make it same size as original image
        frame_ratio = frame_ratio/255 # might need to scale differently
        frame_tmp = cv2.resize(frame_ratio, (width,height),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
        cv2.putText(frame_tmp,"Frame:{}".format(counter), textLocation0, font, fontScale, fontColor, lineType)
        cv2.imshow(ratioed_window_name, frame_tmp)

        # # Bandpasse Image, is difference between 0.5Hz and 10Hz
        # data_bandpass   = data_lowpass_h - data_lowpass_l

        # # transfrom bandpassed image to enhance small changes and to be in 0...1 range
        # data_bandpass_tmp = displaytrans(data_bandpass)        
        # data_bandpass_tmp = cv2.resize(data_bandpass_tmp, (width,height))
        # # Display bandpassed image
        # cv2.imshow(processed_window_name, data_bandpass_tmp)

        if cv2.waitKey(1) & 0xFF == ord('q'): stop = True

        last_display = current_time
        counter += 1

    if (current_time - last_time) >= 5.0: # framearray rate every 5 secs
        logger.log(logging.INFO, "Bin:{}".format(bin_time/5.0))
        bin_time = 0
        last_time = current_time

# Cleanup
camera.stop()
#processor_l.stop()
#processor_h.stop()
cv2.destroyAllWindows()
