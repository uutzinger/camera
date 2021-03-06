configs = {
    ##############################################
    # Camera Settings
    # 320x240 90fps
    # 640x480 90fps
    # 1280x720 60fps
    # 1920x1080 4.4fps
    # 3280x2464 2.8fps
    # auto exposure 0 or 1
    ##############################################
    'camera_res'      : (1280, 720),    # any amera: Camera width & height
    'exposure'        : 10000,          # any camera: -1,0 = auto, 1...max=frame interval
                                        # picamera microseconds
    'autoexposure'    : 0,              # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 60,             # any camera: 1/10, 15, 30, 40, 90, 120 overlocked
    'fourcc'          : 'YU12',         # cv2 camera only: MJPG, YUY2, for ELP camera https://www.fourcc.org/
                                        # CAP_PROP_FOURCC 
    'buffersize'      : 4,              # default is 4 for V4L2, max 10, 
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
    'displayfps'       : 30              # frame rate for display server
    }
