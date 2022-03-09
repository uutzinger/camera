import numpy as np
from numpy.lib.stride_tricks import as_strided
import time
import cv2
from numba import jit, prange
from scipy import signal
from scipy import ndimage

# 840 * 1680
# BINNING          10   8    4    3    2
# ======================================
# shape            69  72   87  108  139
# jit shape        62  59   42   37   47
# take             28  35   68   95  159
# stride          288 478 1750 3056 7011
# cv2 filter2d    133 109   42   31   21
# signal con      378 354  419  371  827
# ndimage con     592 363  106   80   52

# 540x720      54x72 108x144  135x180 180x240 270x360
# BINNING         10       5        4       3       2
# ===================================================
# shape          12.6   10.2     11.7    14.1    19.2
# jit shape       3.5    4.2      4.6     5.1     6.4 
# jit pa shape    1.9    3.0      3.8     4.5     6.4
# take            4.3    7.8     10.3    13.3    21.4
# sum 8           0.8    0.8      1.0     1.0     2.4
# sum 8 direct    0.22   0.27     0.3     0.5     0.9
# cv2 filtert 2d 14.8    5.0      2.9     1.9     1.5
# signal con     77.0   76.7     78.2    77.8   120.2
# ndimage con    82.5   19.8     13.1     9.1     7.4
# stride         41.8  173.1    210.2   366.7   857.9

height = 2*3*2*5*2*7
width = 2*3*2*5*2*7*2

def checkerboard(N,n):
    """N: size of board; n=size of each square; N/(2*n) must be an integer """
    if (N%(2*n)):
        print('Error: N/(2*n) must be an integer')
        return False
    a = np.concatenate((np.zeros(n),np.ones(n)))
    b=np.pad(a,int((N**2)/2-n),'wrap').reshape((N,N))
    return (b+b.T==1).astype(int)

#data_2d = checkerboard(height,60).astype('uint16')
#data_2d = np.concatenate((data_2d,np.ones(np.shape(data_2d))), axis=1)
#data_3d = np.dstack((data_2d,data_2d,data_2d))
#cv2.imshow('Original', 255*data_3d.astype('uint8'))
#cv2.waitKey(1)

bin_x = 10
bin_y = 10
height = 540
width = 720
data_3d = 2*np.ones((height, width, 3), 'uint8')
data_2d = 2*np.ones((height, width), 'uint8')

def bin_shape(arr, binx, biny):
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    m,n,o = np.shape(arr)
    shape = (m//binx, binx, n//biny, biny, o)
    return arr.reshape(shape).sum(3).sum(1)

@jit(nopython=True, fastmath=True)
def bin_shapejit(arr, binx, biny):
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    m,n,o = np.shape(arr)
    shape = (m//binx, binx, n//biny, biny, o)
    return arr.reshape(shape).sum(3).sum(1)

# does not work
# @jit(nopython=True, fastmath=True)
def bin_shapejitpara(arr, binx, biny):
    # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    m,n,o = np.shape(arr)
    out = np.empty((m//binx,n//biny,3), dtype='uint16')
    shape = (m//binx, binx, n//biny, biny)
    for i in prange(o):
        out[:,:,i] = arr[:,:,i].reshape(shape).sum(3).sum(1) 
    return out

def bin_stride(arr,bin_x,bin_y):
    # https://stackoverflow.com/questions/40097213/how-do-i-median-bin-a-2d-image-in-python
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
    m,n = np.shape(arr)
    strided_reshape = as_strided(arr,shape=(m//bin_x,n//bin_y,bin_x,bin_y), strides = arr.itemsize*np.array([bin_x * n, bin_y, n, 1]))
    return np.array([np.sum(col) for row in strided_reshape for col in row]).reshape(m//bin_x,n//bin_y)

@jit(nopython=True, fastmath=True, parallel=True)
def bin_sum8_super(arr_in, binx, biny):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//binx,n,o), dtype='uint16')
    arr_out = np.empty((m//binx,n//biny,o), dtype='uint16')

    if binx == 2:
        for i in prange(m//2):
            arr_tmp[i,:,:] =  arr_in[i*2,:,:] +  arr_in[i*2+1,:,:]
    elif binx == 3:
        for i in prange(m//3):
            arr_tmp[i,:,:] =  arr_in[i*3,:,:] +  arr_in[i*3+1,:,:] +  arr_in[i*3+2,:,:]
    elif binx == 4:
        for i in prange(m//4):
            arr_tmp[i,:,:] =  arr_in[i*4,:,:] +  arr_in[i*4+1,:,:] +  arr_in[i*4+2,:,:] +  arr_in[i*4+3,:,:]
    elif binx == 5:
        for i in prange(m//5):
            arr_tmp[i,:,:] =  arr_in[i*5,:,:] +  arr_in[i*5+1,:,:] +  arr_in[i*5+2,:,:] +  arr_in[i*5+3,:,:] +  arr_in[i*5+4,:,:]
    elif binx == 6:
        for i in prange(m//6):
            arr_tmp[i,:,:] =  arr_in[i*6,:,:] +  arr_in[i*6+1,:,:] +  arr_in[i*6+2,:,:] +  arr_in[i*6+3,:,:] +  arr_in[i*6+4,:,:] +  arr_in[i*6+5,:,:]
    elif binx == 7:
        for i in prange(m//7):
            arr_tmp[i,:,:] =  arr_in[i*7,:,:] +  arr_in[i*7+1,:,:] +  arr_in[i*7+2,:,:] +  arr_in[i*7+3,:,:] +  arr_in[i*7+4,:,:] +  arr_in[i*7+5,:,:] +  arr_in[i*7+6,:,:]
    elif binx == 8:
        for i in prange(m//8):
            arr_tmp[i,:,:] =  arr_in[i*8,:,:] +  arr_in[i*8+1,:,:] +  arr_in[i*8+2,:,:] +  arr_in[i*8+3,:,:] +  arr_in[i*8+4,:,:] +  arr_in[i*8+5,:,:] +  arr_in[i*8+6,:,:] +  arr_in[i*8+7,:,:]
    elif binx == 9:
        for i in prange(m//9):
            arr_tmp[i,:,:] =  arr_in[i*9,:,:] +  arr_in[i*9+1,:,:] +  arr_in[i*9+2,:,:] +  arr_in[i*9+3,:,:] +  arr_in[i*9+4,:,:] +  arr_in[i*9+5,:,:] +  arr_in[i*9+6,:,:] +  arr_in[i*9+7,:,:] +  arr_in[i*9+8,:,:] 
    elif binx == 10:
        for i in prange(m//10):
            arr_tmp[i,:,:] =  arr_in[i*10,:,:] +  arr_in[i*10+1,:,:] +  arr_in[i*10+2,:,:] +  arr_in[i*10+3,:,:] +  arr_in[i*10+4,:,:] +  arr_in[i*10+5,:,:] +  arr_in[i*10+6,:,:] +  arr_in[i*10+7,:,:] +  arr_in[i*10+8,:,:] +  arr_in[i*10+9,:,:]
    else:
        for i in prange(m//binx):
            arr_tmp[i,:,:]   = np.sum(arr_in[i*binx:i*binx+binx:1,:,:],axis=0)

    if biny == 2:
        for j in prange(n//2):
            arr_out[:,j,:] = arr_tmp[:,j*2,:] + arr_tmp[:,j*2+1,:]
    elif biny == 3:
        for j in prange(n//3):
            arr_out[:,j,:] = arr_tmp[:,j*3,:] + arr_tmp[:,j*3+1,:] +  arr_tmp[:,j*3+2,:]
    elif biny == 4:
        for j in prange(n//4):
            arr_out[:,j,:] = arr_tmp[:,j*4,:] + arr_tmp[:,j*4+1,:] +  arr_tmp[:,j*4+2,:] +  arr_tmp[:,j*4+3,:]
    elif biny == 5:
        for j in prange(n//5):
            arr_out[:,j,:] = arr_tmp[:,j*5,:] + arr_tmp[:,j*5+1,:] + arr_tmp[:,j*5+2,:] + arr_tmp[:,j*5+3,:] + arr_tmp[:,j*5+4,:]
    elif biny == 6:
        for j in prange(n//6):
            arr_out[:,j,:] = arr_tmp[:,j*6,:] + arr_tmp[:,j*6+1,:] + arr_tmp[:,j*6+2,:] + arr_tmp[:,j*6+3,:] + arr_tmp[:,j*6+4,:] + arr_tmp[:,j*6+5,:]
    elif biny == 7:
        for j in prange(n//7):
            arr_out[:,j,:] = arr_tmp[:,j*7,:] + arr_tmp[:,j*7+1,:] + arr_tmp[:,j*7+2,:] + arr_tmp[:,j*7+3,:] + arr_tmp[:,j*7+4,:] + arr_tmp[:,j*7+5,:] + arr_tmp[:,j*7+6,:]
    elif biny == 8:
        for j in prange(n//8):
            arr_out[:,j,:] = arr_tmp[:,j*8,:] + arr_tmp[:,j*8+1,:] + arr_tmp[:,j*8+2,:] + arr_tmp[:,j*8+3,:] + arr_tmp[:,j*8+4,:] + arr_tmp[:,j*8+5,:] + arr_tmp[:,j*8+6,:] + arr_tmp[:,j*8+7,:]
    elif biny == 9:
        for j in prange(n//9):
            arr_out[:,j,:] = arr_tmp[:,j*9,:] + arr_tmp[:,j*9+1,:] + arr_tmp[:,j*9+2,:] + arr_tmp[:,j*9+3,:] + arr_tmp[:,j*9+4,:] + arr_tmp[:,j*9+5,:] + arr_tmp[:,j*9+6,:] + arr_tmp[:,j*9+7,:] + arr_tmp[:,j*9+8,:]
    elif biny == 10:
        for j in prange(n//10):
            arr_out[:,j,:] = arr_tmp[:,j*10,:] + arr_tmp[:,j*10+1,:] + arr_tmp[:,j*10+2,:] + arr_tmp[:,j*10+3,:] + arr_tmp[:,j*10+4,:] + arr_tmp[:,j*10+5,:] + arr_tmp[:,j*10+6,:] + arr_tmp[:,j*10+7,:] + arr_tmp[:,j*10+8,:] + arr_tmp[:,j*10+9,:]
    else:
        for j in prange(n//bin_y):
            arr_out[:,j,:] = np.sum(arr_tmp[:,j*biny:j*biny+biny:1,:],axis=1)
    return arr_out

@jit(nopython=True, fastmath=True, parallel=True)
def bin_sum8(arr_in, binx, biny):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//binx,n,o), dtype='uint16')
    arr_out = np.empty((m//binx,n//biny,o), dtype='uint16')
    for i in prange(m//binx):
        arr_tmp[i,:,:] = np.sum(arr_in[i*binx:i*binx+binx:1,:,:],axis=0)
    for j in prange(n//bin_y):
        arr_out[:,j,:] = np.sum(arr_tmp[:,j*biny:j*biny+biny:1,:],axis=1)
    return arr_out

@jit(nopython=True, fastmath=True, parallel=True)
def bin2_sum8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//2,n,o), dtype='uint16')
    arr_out = np.empty((m//2,n//2,o), dtype='uint16')
    for i in prange(m//2):
        arr_tmp[i,:,:] =  arr_in[i*2,:,:] +  arr_in[i*2+1,:,:]
    for j in prange(n//2):
        arr_out[:,j,:] = arr_tmp[:,j*2,:] + arr_tmp[:,j*2+1,:]
    return arr_out

@jit(nopython=True, fastmath=True, parallel=True)
def bin3_sum8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//3,n,o), dtype='uint16')
    arr_out = np.empty((m//3,n//3,o), dtype='uint16')
    for i in prange(m//3):
        arr_tmp[i,:,:] =  arr_in[i*3,:,:] +  arr_in[i*3+1,:,:] +  arr_in[i*3+2,:,:]
    for j in prange(n//3):
        arr_out[:,j,:] = arr_tmp[:,j*3,:] + arr_tmp[:,j*3+1,:] +  arr_tmp[:,j*3+2,:]
    return arr_out

@jit(nopython=True, fastmath=True, parallel=True)
def bin4_sum8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//4,n,o), dtype='uint16')
    arr_out = np.empty((m//4,n//4,o), dtype='uint16')
    for i in prange(m//4):
        arr_tmp[i,:,:] =  arr_in[i*4,:,:] +  arr_in[i*4+1,:,:] +  arr_in[i*4+2,:,:] +  arr_in[i*4+3,:,:]
    for j in prange(n//4):
        arr_out[:,j,:] = arr_tmp[:,j*4,:] + arr_tmp[:,j*4+1,:] +  arr_tmp[:,j*4+2,:] +  arr_tmp[:,j*4+3,:]
    return arr_out

@jit(nopython=True, fastmath=True, parallel=True)
def bin5_sum8(arr_in):
    m,n,o   = np.shape(arr_in)
    arr_tmp = np.empty((m//5,n,o), dtype='uint16')
    arr_out = np.empty((m//5,n//5,o), dtype='uint16')
    for i in prange(m//5):
        arr_tmp[i,:,:] =  arr_in[i*5,:,:] +  arr_in[i*5+1,:,:] +  arr_in[i*5+2,:,:] +  arr_in[i*5+3,:,:] +  arr_in[i*5+4,:,:]
    for j in prange(n//5):
        arr_out[:,j,:] = arr_tmp[:,j*5,:] + arr_tmp[:,j*5+1,:] + arr_tmp[:,j*5+2,:] + arr_tmp[:,j*5+3,:] + arr_tmp[:,j*5+4,:]
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

@jit(nopython=True, fastmath=True, parallel=True)
def bin_stridejit(arr,bin_x,bin_y):
    # https://stackoverflow.com/questions/40097213/how-do-i-median-bin-a-2d-image-in-python
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html
    m,n = np.shape(arr)
    strided_reshape = as_strided(arr,shape=(m//bin_x,n//bin_y,bin_x,bin_y), strides = arr.itemsize*np.array([bin_x * n, bin_y, n, 1]))
    return np.array([np.sum(col) for row in strided_reshape for col in row]).reshape(m//bin_x,n//bin_y)

def bin_take(arr, bin_x, bin_y):
    # https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730
    m,n,o = np.shape(arr)
    t1 = np.array([np.sum(np.take(arr, range(i*bin_x,i*bin_x+bin_x),axis=0),axis=0) for i in range(m//bin_x)])
    t2 = np.array([np.sum(np.take(t1,  range(j*bin_y,j*bin_y+bin_y),axis=1),axis=1) for j in range(n//bin_y)])
    return np.transpose(t2, axes=(1,0,2))

#@jit(nopython=True, fastmath=True, parallel=True)
def bin_takejit(arr, bin_x, bin_y):
    # https://stackoverflow.com/questions/21921178/binning-a-numpy-array/42024730#42024730
    m,n,o = np.shape(arr)
    t1 = np.array([np.sum(np.take(arr, range(i*bin_x,i*bin_x+bin_x),axis=0),axis=0) for i in range(m//bin_x)])
    t2 = np.array([np.sum(np.take(t1,  range(j*bin_y,j*bin_y+bin_y),axis=1),axis=1) for j in range(n//bin_y)])
    return np.transpose(t2, axes=(1,0,2))
   
data_binned = bin_shape(data_3d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_shape(data_3d,bin_x, bin_y)
print("Color image binning with shape took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Shape', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin_shapejit(data_3d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_shapejit(data_3d,bin_x, bin_y)
print("Color image binning with jit shape took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('SSUM8hape JIT', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin_shapejitpara(data_3d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_shapejitpara(data_3d,bin_x, bin_y)
print("Color image binning with jit parallel shape took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('SSUM8hape JIT', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin_take(data_3d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_take(data_3d,bin_x, bin_y)
print("Color image binning with take took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin_takejit(data_3d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_takejit(data_3d,bin_x, bin_y)
print("Color image binning with jit take took: {:.4f}s".format((time.perf_counter()-start_time)/10.))

data_binned = bin_sum8(data_3d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_sum8(data_3d,bin_x, bin_y)
print("Color image binning with sum 8 took: {:.5f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin2_sum8(data_3d)
start_time = time.perf_counter()
for i in range(10): data_binned = bin2_sum8(data_3d)
print("Color image binning by 2 with sum 8 took: {:.5f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin3_sum8(data_3d)
start_time = time.perf_counter()
for i in range(10): data_binned = bin3_sum8(data_3d)
print("Color image binning by 3 with sum 8 took: {:.5f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin4_sum8(data_3d)
start_time = time.perf_counter()
for i in range(10): data_binned = bin4_sum8(data_3d)
print("Color image binning by 4 with sum 8 took: {:.5f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin5_sum8(data_3d)
start_time = time.perf_counter()
for i in range(10): data_binned = bin5_sum8(data_3d)
print("Color image binning by 5 with sum 8 took: {:.5f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin10_sum8(data_3d)
start_time = time.perf_counter()
for i in range(10): data_binned = bin10_sum8(data_3d)
print("Color image binning by 10 with sum 8 took: {:.5f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('Take', data_binned.astype('uint8'))
#cv2.waitKey(1)


kernel  = np.ones((bin_x, bin_y),'uint16')
data_binned = cv2.filter2D(data_3d,-1,kernel)
start_time = time.perf_counter()
for i in range(10): data_binned = cv2.filter2D(data_3d,-1,kernel)
print("Color image binning with opencv filter2D took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('CV2filter2D', data_binned.astype('uint8'))
#cv2.waitKey(1)

kernel  = np.ones((bin_x, bin_y,1),'uint16')
data_binned = signal.convolve(data_3d, kernel, mode='same', method='auto')
start_time = time.perf_counter()
for i in range(10): data_binned = signal.convolve(data_3d, kernel, mode='same', method='auto')
print("Color image binning with scipy.signal convolve2d took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('SignalConvolve', data_binned.astype('uint8'))
#cv2.waitKey(1)

kernel  = np.ones((bin_x, bin_y,1),'uint16')
data_binned = ndimage.convolve(data_3d, kernel)
start_time = time.perf_counter()
for i in range(10): data_binned = ndimage.convolve(data_3d, kernel)
print("Color image binning with scipy.ndimage convolve took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('ndimageConvolve', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = cv2.filter2D(data_2d,-1,kernel)
start_time = time.perf_counter()
for i in range(10): data_binned = cv2.filter2D(data_2d,-1,kernel)
print("B/W image binning with opencv filter2D: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('OpenCVfilter2DBW', data_binned.astype('uint8'))
#cv2.waitKey(1)

kernel  = np.ones((bin_x, bin_y),'uint16')
data_binned = signal.convolve(data_2d, kernel, mode='same', method='auto')
start_time = time.perf_counter()
for i in range(10): data_binned = signal.convolve(data_2d, kernel, mode='same', method='auto')
print("B/W image binning with scipy.signal convolve2d took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('SignalConvolveBW', data_binned.astype('uint8'))
#cv2.waitKey(1)

kernel  = np.ones((bin_x, bin_y),'uint16')
data_binned = ndimage.convolve(data_2d, kernel)
start_time = time.perf_counter()
for i in range(10): data_binned = ndimage.convolve(data_2d, kernel)
print("B/W image binning with scipy.ndimage convolve took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
#cv2.imshow('ndimageConvolveBW', data_binned.astype('uint8'))
#cv2.waitKey(1)

data_binned = bin_stride(data_2d,bin_x, bin_y)
start_time = time.perf_counter()
for i in range(10): data_binned = bin_stride(data_2d.astype('int64'),bin_x, bin_y)
print("B/W image binning with stride took: {:.4f}s".format((time.perf_counter()-start_time)/10.))
cv2.destroyAllWindows()
#cv2.imshow('Stride', data_binned.astype('uint8'))
#cv2.waitKey(1)

#data_binned = bin_stridejit(data_2d,bin_x, bin_y)
#start_time = time.perf_counter()
#for i in range(10): data_binned = bin_stridejit(data_2d.astype('int64'),bin_x, bin_y)
#print("B/W image binning with stride took: {:.4f}s".format((time.perf_counter()-start_time)/10.))

#cv2.destroyAllWindows()
