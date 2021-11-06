configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res'      : (1920, 1080),   # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
                                        # 1920x1980 30fps
                                        # 1280x720 60fps
                                        # 640,480, 120fps
    'exposure'        : -6,             # camera specific e.g. -5 =(2^-5)=1/32, 0 = auto, 1...max=frame interval in microseconds
    'autoexposure'    : 3.0,            # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 30,             # 120fps only with MJPG fourcc
    'fourcc'          : "MJPG",         # cv2 camera only: MJPG, YUY2, YUYV
    'buffersize'      : -1,             # default is 4 for V4L2, max 10, 
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
    'displayfps'       : 5              # frame rate for display server
    }
