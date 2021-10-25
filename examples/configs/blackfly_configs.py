configs = {
    ##############################################
    # Camera Settings
    # 720 x 540
    # auto exposure 0 or 1
    ##############################################
    'camera_res'      : (720, 540),     # image width & height, can read ROI
    'exposure'        : 1750,           # in microseconds, -1 = autoexposure
    'autoexposure'    : 0,              # 0,1
    'fps'             : 500,            # 
    'binning'         : (1,1),          # 1,2 or 4
    'offset'          : (0,0),          #
    'adc'             : 8,              # 8,10,12,14 bit
    'trigout'         : 2,              # -1 no trigger output, 
                                        # line 1 has opto isolator but requires pullup to 3V
                                        # line 2 has not isolation and takes 4-10us for a transition
    'ttlinv'          : True,           # inverted logic levels are best
    'trigin'          : -1,             # -1 use software, otherwise hardware
    ##############################################
    # Target Display
    ##############################################
    'output_res'      : (-1, -1),       # Output resolution, -1 = do not change
    'flip'            : 0,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'displayfps'       : 50             # frame rate for display, usually we skip frames for display but record at full camera fps
    }
