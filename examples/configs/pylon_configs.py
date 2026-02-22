configs = {
    ##############################################
    # Camera settings
    ##############################################
    "camera_res": (720, 540),          # image width & height (ROI)
    "stream_policy": "default",        # 'default'|'max_fps'|'auto'
    "exposure": 1750,                  # microseconds, -1 leaves camera auto behavior
    "autoexposure": 0,                 # -1 unchanged, 0 off, 1 on
    "fps": 60,                         # requested acquisition frame rate
    "binning": (1, 1),                 # sensor binning (h, v)
    "offset": (0, 0),                  # ROI offset (x, y)
    "adc": -1,                         # keep camera default unless model supports explicit setting
    "pixel_format": "",                # optional override, e.g. 'Mono8', 'Mono16', 'BayerRG8'
    "convert_format": "native",        # 'native'|'Mono8'|'Mono16'|'BGR8'|'RGB8'
    "trigout": -1,                     # -1 no trigger output, >0 uses matching line
    "ttlinv": False,                   # invert trigger polarity
    "trigin": -2,                      # -2 leave default, -1 software trigger, >0 hardware line
    "capture_timeout_ms": 100,         # RetrieveResult timeout in ms
    "buffersize": 8,                   # FrameBuffer capacity
    "buffer_overwrite": True,          # overwrite old frames if buffer full
    ##############################################
    # Target display
    ##############################################
    "output_res": (-1, -1),            # Output resolution, -1 = do not change
    "flip": 0,                         # 0=norotation, 1=ccw90, 2=180, 3=cw90, 4=hflip, 5=diag, 6=vflip, 7=other diag
    "displayfps": 30,                  # display rate; capture can run faster
}
