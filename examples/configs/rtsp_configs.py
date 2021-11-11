configs = {
    ##############################################
    # Camera Settings
    ##############################################
    'camera_res'      : (1280, 720 ),   # any amera: Camera width & height
    'fps'             : 30,             # any camera: 1/10, 15, 30, 40, 90, 120, 180
    'fourcc'          : -1,             # not used
    'buffersize'      : -1,             # not used 
    'fov'             : 77,             # camera lens field of view in degress
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
