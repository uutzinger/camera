##########################################################################
# Testing of capture to RTP using multiprocessing (cv2CaptureProc + rtpServerProc)
##########################################################################

import cv2
import logging
import platform
import time


def main():

    camera_index   = 0  # default camera starts at 0 by operating system
    window_name    = 'Camera'
    font           = cv2.FONT_HERSHEY_SIMPLEX
    textLocation   = (10,20)
    fontScale      = 1
    fontColor      = (255,255,255)
    lineType       = 2
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

    from configs.eluk_configs import configs as configs

    if configs['displayfps'] >= configs['fps']:
        rtp_interval = 1.0/configs['fps']
        rtp_fps = configs['fps']
        rtp_slowdown = False
    else:
        rtp_interval = 1.0/configs['displayfps']
        rtp_fps = configs['displayfps']
        rtp_slowdown = True

    rtp_size = configs['output_res']
    if rtp_size[0] <= 0 or rtp_size[1] <= 0:
        rtp_size = configs['camera_res']

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("cv2_capture_send2rtp_process")

    from camera.streamer.rtpserverProc import rtpServerProc
    print("Starting rtp Server")
    rtp = rtpServerProc(
        resolution = rtp_size,
        fps        = rtp_fps,
        host       = '127.0.0.1',
        port       = 554,
        bitrate    = 2048,
        color      = True,
        gpu        = False,
    )
    rtp.start()

    plat = platform.system()
    if plat == 'Linux' and platform.machine() == "aarch64":
        from camera.capture.nanocapture import nanoCapture
        camera = nanoCapture(configs, camera_index)
    else:
        from camera.capture.cv2captureProc import cv2CaptureProc
        camera = cv2CaptureProc(configs, camera_index)

    print("Getting Images")
    camera.start()

    last_time = time.time()
    want_refresh = False

    print('To test RTP start: gst-launch-1.0 -v udpsrc port=554 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false')

    while True:
        (frame_time, frame) = camera.capture.get(block=True, timeout=None)

        if rtp_slowdown:
            current_time = time.time()
            if (current_time - last_time) >= rtp_interval:
                want_refresh = True
                last_time = current_time

        if want_refresh or not rtp_slowdown:
            frame_rtp = frame.copy()
            cv2.putText(frame_rtp, "Capture FPS:{} [Hz]".format(camera.measured_fps), textLocation, font, fontScale, fontColor, lineType)

            try:
                rtp.queue.put_nowait((frame_time, frame_rtp))
            except Exception:
                logger.log(logging.WARNING, "Status:rtp Queue is full!")

            cv2.imshow(window_name, frame_rtp)
            want_refresh = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                    break
            except Exception:
                break

            try:
                (level, msg) = rtp.log.get_nowait()
                logger.log(level, "Status:{}".format(msg))
            except Exception:
                pass

    camera.stop()
    rtp.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
