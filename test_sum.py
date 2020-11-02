import numpy as np
import time

total = 0.0
for i in range(0,100):
    A=np.random.randint(0,255,(720,540),dtype=np.uint8)
    tic=time.perf_counter()
    stat=np.sum(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("NumPy  uint8 {}".format(total/100.0))
#print(stat)
total=0.0
for i in range(0,100):
    A=np.random.randint(0,255,(720,540),dtype=np.uint16)
    tic=time.perf_counter()
    stat=np.sum(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("NumPy uint16 {}".format(total/100.0))
#print(stat)
total=0.0
for i in range(0,100):
    A=np.random.randint(0,255,(720,540),dtype=np.uint32)
    tic=time.perf_counter()
    stat=np.sum(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("NumPy uint32 {}".format(total/100.0))
#print(stat)
total=0.0
for i in range(0,100):
    A=np.random.rand(720,540)
    tic=time.perf_counter()
    stat=np.sum(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("NumPy  float {}".format(total/100.0))
#print(stat)
print('')

from PIL import ImageStat
from PIL import Image

total = 0.0
for i in range(0,100):
    A=Image.fromarray(np.random.randint(0,255,(720,540),dtype=np.uint8))
    tic=time.perf_counter()
    stat=ImageStat.Stat(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("PIL  uint8 {}".format(total/100.0))
#print(stat.sum)
total=0.0
for i in range(0,100):
    A=Image.fromarray(np.random.randint(0,255,(720,540),dtype=np.uint16))
    tic=time.perf_counter()
    stat=ImageStat.Stat(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("PIL uint16 {}".format(total/100.0))
#print(stat.sum)
total=0.0
for i in range(0,100):
    A=Image.fromarray(np.random.randint(0,255,(720,540),dtype=np.uint32))
    tic=time.perf_counter()
    stat=ImageStat.Stat(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("PIL uint32 {}".format(total/100.0))
#print(stat.sum)
total=0.0
for i in range(0,100):
    A=Image.fromarray(np.random.rand(720,540))
    tic=time.perf_counter()
    stat=ImageStat.Stat(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("PIL  float {}".format(total/100.0))
#print(stat.sum)
print('')

import cv2

total = 0.0
for i in range(0,100):
    A=(np.random.randint(0,255,(720,540),dtype=np.uint8))
    tic=time.perf_counter()
    stat=cv2.sumElems(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("CV2  uint8 {}".format(total/100.0))
#print(stat)
total=0.0
for i in range(0,100):
    A=(np.random.randint(0,255,(720,540),dtype=np.uint16))
    tic=time.perf_counter()
    stat=cv2.sumElems(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("CV2 uint16 {}".format(total/100.0))
#print(stat)
total=0.0
for i in range(0,100):
    A=(np.random.rand(720,540))
    tic=time.perf_counter()
    stat=cv2.sumElems(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("CV2  float {}".format(total/100.0))
#print(stat)
print('')

total = 0.0
for i in range(0,100):
    A=(np.random.randint(0,255,(180,135),dtype=np.uint8))
    # Ah = cv2.resize(A,dsize=(180,135),interpolation = cv2.INTER_NEAREST)
    # Ah=A[0::4,0::4]
    tic=time.perf_counter()
    stat=cv2.sumElems(A)
    toc=time.perf_counter()
    total = total + (toc-tic)
print("CV2 resize 0.25 uint8 {}".format(total/100.0))

