import logging
import time
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# define backend
# pip3 install  wxPython
# matplotlib.use(wxAgg)

logging.basicConfig(level=logging.DEBUG)

# create figure
fig, ax = plt.subplots()

print("Display Capture")

print("Getting Dat")

last_fps_time = time.time()
while(True):

    # Plot Cross Hair
    
    # Plot Target Center
    
    # Plot Corners`
    ax.plot(corners_x, corners_y, {'marker': 'o'})
    ax.grid()
    ax.set_xlim([0,320])
    ax.set_ylim([0,240])
    

    plt.show()
