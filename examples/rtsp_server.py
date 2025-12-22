##########################################################################
# Example: RTSP server (rtspServer)
##########################################################################

import logging
import time
import numpy as np

from camera.streamer.rtspserver import rtspServer


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("rtsp_server")

    fps = 30
    width, height = 640, 480

    # NOTE:
    # - Default RTSP port for many IP cameras is 554, but binding to 554 on Linux
    #   may require elevated privileges.
    # - This example uses 8554.
    host = "0.0.0.0"
    port = 8554
    uri = "/test"

    server = rtspServer(
        resolution=(width, height),
        fps=fps,
        bitrate=2048,
        host=host,
        port=port,
        stream_uri=uri,
    )

    logger.info("Starting RTSP server")
    server.start()

    logger.info("Stream URL: rtsp://127.0.0.1:%d%s", port, uri)

    interval = 1.0 / fps
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        while True:
            # simple moving pattern
            t = int(time.time() * 10) % width
            frame[:] = 0
            frame[:, t : t + 10, :] = 255

            if not server.queue.full():
                server.queue.put_nowait((time.time(), frame))

            while not server.log.empty():
                level, msg = server.log.get_nowait()
                logger.log(level, msg)

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        server.join(timeout=2.0)


if __name__ == "__main__":
    main()
