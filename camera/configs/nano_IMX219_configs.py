configs = {
    ##############################################
    # Camera Settings
    # 320x240 90fps
    # 640x480 90fps 
    # 1280x720 60fps (60/120fps)
    # 1920x1080 4.4fps (30fps)
    # 3280x2464 2.8fps (21fps)
    # auto exposure 0 or 1
    ##############################################
    'camera_res'      : (1280, 720),    # width & height
    'exposure'        : 10000,          # microseconds, internally converted to nano seconds
                                        # <= 0 autoexposure
    'fps'             : 60,             # 
    ##############################################
    # Target Display
    ##############################################
    'output_res'      : (-1, -1),       # Output resolution 
    'flip'            : 0,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'serverfps'       : 30
    }

##################################################
# Capture Options for Sony IX219 CSI camera
##################################################
# v4l2-ctl -L
# v4l2-ctl --list-formats-ext
