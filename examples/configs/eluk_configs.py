configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res'      : (1280, 720 ),   # any amera: Camera width & height
                                        # 1280x720 30fps
                                        # 640x480 30fps
                                        # 640x360 30fps
                                        # 352x288 30fps
                                        # 320x240 30fps
                                        # 176x144 30fps
                                        # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
    'exposure'        : -2,             # any camera: -1,0 = auto, 1...max=frame interval
                                        # picamera microseconds
                                        # opencv CAP_PROP_EXPOSURE
    'autoexposure'    : 0.0,            # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 30,             # any camera: 1/10, 15, 30, 40, 90, 120, 180
    'fourcc'          : -1,             # cv2 camera reports 22
                                        # Laptop Windows -1
    'buffersize'      : -1,             # default is 4 for V4L2, max 10, 
                                        # Laptop: -1
    'gain'            : 4,              # Sensor readout gain
    'autowb'          : -1,             # enable auto white balancing
    'wb_temp'         : -1,             # white balance temperature
    'settings'        : 0,              # open camera settings window
    ##############################################
    # Target Recognition
    ##############################################
    'fov'             : 77,             # camera lens field of view in degress
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
    'displayfps'       : 30             # frame rate for display server
    }

# BACKLIGHT 2.0, 1.0, 0.0
# BACKEND 700, cannot change
# SETTINGS 0.0 anyvalyue opens settings window