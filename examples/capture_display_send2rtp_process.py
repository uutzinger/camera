##########################################################################
# Testing of capture to rtp 
##########################################################################
# % CPU usage
##########################################################################

import cv2
import logging, time, platform

def main():

    # Setting up display
    camera_index   = 0  # default camera starts at 0 by operating system
    window_name    = 'Camera'
    font           = cv2.FONT_HERSHEY_SIMPLEX
    textLocation   = (10,20)
    fontScale      = 1
    fontColor      = (255,255,255)
    lineType       = 2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

    from examples.configs.eluk_configs import configs as configs
    # -Dell Inspiron 15 internal camera
    #   from configs.dell_internal_configs  import configs as configs
    # -Eluktronics Max-15 internal camera
    # -Generic webcam
    #   from configs.generic_1080p import configs as configs
    # -Nano Jetson IMX219 camera
    #   from configs.nano_IMX219_configs  import configs as configs
    # -Raspberry Pi v1 & v2 camera
    #   from configs.raspi_v1module_configs  import configs as configs
    #   from configs.raspi_v2module_configs  import configs as configs
    # -ELP 
    #   from configs.ELP1080p_configs  import configs as configs
    # -FLIR Lepton 3.5
    #   from configs.FLIRlepton35 import confgis as configs

    if configs['displayfps'] >= configs['fps']:
        rtp_interval = 1.0/configs['fps']
        rtp_fps = configs['fps']
        rtp_slowdown = False
    else:
        rtp_interval = 1.0/configs['displayfps']
        rtp_fps = configs['displayfps']
        rtp_slowdown = True

    rtp_size = configs['output_res']
    if rtp_size[0]<=0 or rtp_size[1]<=0: 
        rtp_size =  configs['camera_res']

    # Setting up logging
    logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
    logger = logging.getLogger("Capture2RTP")

    # Setting up RTP
    from camera.streamer.rtpserver_process import rtpServer
    print("Starting rtp Server")
    rtp = rtpServer(
        resolution = rtp_size, 
        fps        = rtp_fps, 
        host       = '127.0.0.1', 
        port       = 554, 
        bitrate    = 2048,
        color      = True, 
        gpu        = False)
    rtp.start()

    # Create camera interface based on computer OS you are running
    # Works for Windows, Linux, MacOS
    plat = platform.system()
    if plat == 'Linux' and platform.machine() == "aarch64": # this is jetson nano for me
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    else:
        from camera.capture.cv2capture_process import cv2Capture
        camera = cv2Capture(configs, camera_index)

    print("Getting Images")
    camera.start()

    # Initialize Variables
    last_time = time.time()

    print('To test RTP start: gst-launch-1.0 -v udpsrc port=554 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false')

    while True:
        # Wait for new image
        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if rtp_slowdown:
            current_time = time.time()
            if (current_time - last_time) >= rtp_interval:
                want_refresh = True
                last_time = current_time

        # Display and transmit
        if want_refresh or not rtp_slowdown:
            # Annotate image
            frame_rtp=frame.copy()
            cv2.putText(frame_rtp,"Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation, font, fontScale, fontColor, lineType)
            # Transmit image
            try:    rtp.queue.put_nowait((frame_time, frame_rtp)) 
            except: logger.log(logging.WARNING, "Status:rtp Queue is full!")
            # Show the captured image
            cv2.imshow(window_name, frame_rtp)
            want_refresh = False

            # Quit the program if users enters q
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Log messages from the process
            try: 
                (level, msg)=rtp.log.get_nowait()
                logger.log(level, "Status:{}".format(msg))
            except: pass

    # Finish
    camera.stop()
    rtp.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()