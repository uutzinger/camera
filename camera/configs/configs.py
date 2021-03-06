configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res'      : (2592, 1944 ),  # any amera: Camera width & height
                                        # CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
    'exposure'        : 1000,           # any camera: -1,0 = auto, 1...max=frame interval
                                        # picamera microseconds
                                        # opencv CAP_PROP_EXPOSURE
    'autoexposure'    : 0,              # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
    'fps'             : 120,            # any camera: 1/10, 15, 30, 40, 90, 120 overlocked, 180?
    'fourcc'          : 'YU12',         # cv2 camera only: MJPG, YUY2, for ELP camera https://www.fourcc.org/         CAP_PROP_FOURCC 
                                        # Laptop Windows -1
    'buffersize'      : 4,              # default is 4 for V4L2, max 10, 
                                        # Laptop: -1
    'rtsp'            : 'rtsp://admin:Password@192.168.0.200:554/',
                                        # port 1181 for opsi raspberrypi server (opensight)
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

##################################################
# To identify Capture Options in Linux
##################################################
# v4l2-ctl -L
# v4l2-ctl --list-formats-ext
