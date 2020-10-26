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

# Network Table attach
NetworkTables.initialize(server="10.41.83.2")
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

sd = NetworkTables.getTable("SmartDashboard")
ll = NetworkTables.getTable("limelight")

last_fps_time = time.time()
while(True):

    robotTime = sd.getNumber("dsTime", -1))

    # Plot Cross Hair
    # at center of graph
    
    # Plot Target Center
    tv=llt.getNumber('tv')
    tx=ll.getNumber('tx')
    ty=ll.getNumber('ty')
    ax.plot(tx, ty, {'marker': 'o'})
    
    # Plot Corners`
    corner_x=ll.getNumber('tcornx')
    corner_y=ll.getNumber('tcorny')
    
    ax.plot(corners_x, corners_y, {'marker': 'o'})
    
    ax.grid()
    ax.set_xlim([0,320])
    ax.set_ylim([0,240])
    
    plt.show()

