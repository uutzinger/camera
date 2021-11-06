# This is to figure out if we need
# image_stack[numbr,image] or image_stack[image,number]
#
import time
import numpy as np
data_cube1 = np.random.randint(0, 255, (540, 720, 14), 'uint8')
data_cube2 = np.random.randint(0, 255, (14, 540, 720), 'uint8')
frame_idx = 0 
counts = 0
assignment_time1 = 0.0
assignment_time2 = 0.0
while True:
    img = np.random.randint(0, 255, (540, 720), 'uint8')
    before_time1 = time.perf_counter()
    data_cube1[:,:,frame_idx] = img # 23ms
    after_time1 = time.perf_counter()
    before_time2 = time.perf_counter()
    data_cube2[frame_idx,:,:] = img # 2ms
    after_time2 = time.perf_counter()
    assignment_time1 = assignment_time1 + (after_time1 - before_time1)
    assignment_time2 = assignment_time2 + (after_time2 - before_time2)
    frame_idx += 1
    if frame_idx >= 14:
        frame_idx = 0
        counts += 1
        print("Array assignment time method1:{}s".format(assignment_time1/counts))
        print("Array assignment time method2:{}s".format(assignment_time2/counts))
