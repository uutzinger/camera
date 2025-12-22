###############################################################################
# RTP point to point server
# 
# Create Stream with 
#    rtp = rtpServer(
#        resolution =(width,height),
#        fps        = fps, 
#        host       = '127.0.0.1', 
#        port       = 554,
#        bitrate    = 2048, 
#        gpu        = False)
#
#    rtp.start()
#    cube = np.random.randint(0, 255, (height, width, depth), 'uint8')
#    rtp.queue.put_nowait(cube)
#
# Urs Utzinger 
#
# Changes:
# 2025 Codereview and cleanup
# 2021 Initial Release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from queue import Queue, Empty

# System
import logging, time
import platform
import subprocess
import os

# Array
import numpy as np

# Open Computer Vision (preferred when built with GStreamer)
import cv2

# FFmpeg helper (bundled binary via pip)
try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None

###############################################################################
# RTP Server
###############################################################################

class rtpServer(Thread):
    """
    Create rtp h264 video network stream
    """

    # Initialize the RTP Thread
    def __init__(self, 
        resolution: tuple = (320, 240), 
        fps: int     = 16,
        host: str    = '127.0.0.1', 
        port: int    = 554,  
        bitrate: int = 2048, 
        gpu: bool    = False,
        sdp_path: str = None,
        payload_type: int = 96,
        queue_size: int = 32,
    ):

        super().__init__(daemon=True)

        self._STOP_ITEM = (None, None)

        # Threading Queue, Logs
        self.queue   = Queue(maxsize=queue_size)
        self.log     = Queue(maxsize=32)
        self.stopped = True

        # populate desired settings from configuration file or function call
        self._port    = port
        self._host    = host
        self._res     = resolution
        self._fps     = fps
        self._fourcc  = 0
        self._bitrate = bitrate
        self._isColor = True
        self._gpuavail = gpu
        self._payload_type = int(payload_type)
        self._sdp_path = sdp_path

        self._backend = None
        self.rtp = None
        self.rtp_open = False
        self._ffmpeg_proc = None
        self._ffmpeg_stderr_thread = None

        self._open_stream()

        self.measured_cps = 0.0

    #
    # Thread routines #################################################
    # Start Stop and Update Thread

    def stop(self):
        """Stop the thread (idempotent)."""
        self.stopped = True
        try:
            if not self.queue.full():
                self.queue.put_nowait(self._STOP_ITEM)
        except Exception:
            pass

        try:
            if self.is_alive():
                self.join(timeout=2.0)
        except Exception:
            pass

        self._close_stream()

    def start(self):
        """Start the thread (non-blocking)."""
        if self.is_alive():
            return
        self.stopped = False
        super().start()

    def run(self):
        """Thread entrypoint."""
        if not getattr(self, 'rtp_open', False):
            return
        self.update()

    # After starting the thread, this runs continously
    def update(self):
        """run the thread"""
        last_time = time.time()
        num_frames = 0
        
        while not self.stopped:
            try:
                item = self.queue.get(timeout=0.25)
            except Empty:
                item = None

            if item is None:
                pass
            elif item == self._STOP_ITEM:
                break
            else:
                frame = None
                if isinstance(item, tuple) and len(item) == 2:
                    _, frame = item
                else:
                    frame = item

                if frame is not None:
                    try:
                        self._write_frame(frame)
                        num_frames += 1
                    except Exception as exc:
                        if not self.log.full():
                            self.log.put_nowait((logging.ERROR, f"RTP:Write failed: {exc}"))

            # RTP through put calculation
            current_time = time.time()
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_cps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "RTP:FPS:{}".format(self.measured_cps)))
                last_time = current_time
                num_frames = 0

            self._close_stream()

    def _open_stream(self):
        """Open a sender backend.

        Preferred: OpenCV+GStreamer (cv2.CAP_GSTREAMER) when available.
        Fallback: FFmpeg subprocess (imageio-ffmpeg) on platforms without OpenCV GStreamer.
        """

        self._backend = None
        self.rtp = None
        self.rtp_open = False
        self._ffmpeg_proc = None
        self._ffmpeg_stderr_thread = None

        # Try OpenCV+GStreamer first
        gst = 'appsrc ! videoconvert ! '
        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64':
                gst = (
                    gst
                    + 'omxh264enc control-rate=1 bitrate={:d} preset-level=1 ! '.format(self._bitrate * 1000)
                    + 'video/x-h264,stream-format=(string)byte-stream ! h264parse ! '
                )
            elif platform.machine() in ('armv6l', 'armv7l'):
                gst = (
                    gst
                    + 'omxh264enc control-rate=1 target-bitrate={:d} ! '.format(self._bitrate * 1000)
                    + 'video/x-h264,stream-format=(string)byte-stream ! h264parse ! '
                )
        else:
            if self._gpuavail:
                gst = gst + 'nvh264enc zerolatency=1 rc-mode=vbr max-bitrate={:d} ! '.format(self._bitrate)
            else:
                gst = gst + 'x264enc tune=zerolatency bitrate={:d} speed-preset=superfast ! '.format(self._bitrate)

        gst = gst + 'rtph264pay config-interval=1 pt={:d} ! udpsink host={:s} port={:d}'.format(
            self._payload_type, self._host, self._port
        )

        if not self.log.full():
            self.log.put_nowait((logging.INFO, gst))

        try:
            self.rtp = cv2.VideoWriter(
                gst,
                apiPreference=cv2.CAP_GSTREAMER,
                fourcc=self._fourcc,
                fps=self._fps,
                frameSize=self._res,
                isColor=self._isColor,
            )
            self.rtp_open = bool(self.rtp is not None and self.rtp.isOpened())
            if self.rtp_open:
                self._backend = 'opencv'
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, "RTP:Backend: OpenCV+GStreamer (no SDP file generated)"))
                return
        except Exception:
            self.rtp = None
            self.rtp_open = False

        # Fallback: FFmpeg subprocess
        if imageio_ffmpeg is None:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTP:No OpenCV+GStreamer and imageio-ffmpeg not installed"))
            return

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        width, height = int(self._res[0]), int(self._res[1])

        sdp_path = self._sdp_path
        if not sdp_path:
            sdp_path = f"rtp_{self._port}.sdp"
            self._sdp_path = sdp_path

        # Encode H264 and send RTP
        cmd = [
            ffmpeg_exe,
            '-loglevel', 'error',
            '-hide_banner',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(self._fps),
            '-i', 'pipe:0',
            '-an',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(max(1, int(self._fps))),
            '-b:v', f'{int(self._bitrate)}k',
            '-maxrate', f'{int(self._bitrate)}k',
            '-bufsize', f'{int(self._bitrate) * 2}k',
            '-pix_fmt', 'yuv420p',
            '-payload_type', str(self._payload_type),
            '-f', 'rtp',
            '-sdp_file', str(sdp_path),
            f'rtp://{self._host}:{int(self._port)}',
        ]

        if not self.log.full():
            self.log.put_nowait((logging.INFO, f"RTP:FFmpeg cmd: {' '.join(cmd)}"))
            self.log.put_nowait((logging.INFO, "RTP:Backend: FFmpeg (SDP file generated)"))
            self.log.put_nowait((logging.INFO, f"RTP:SDP written to: {os.path.abspath(sdp_path)}"))

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._start_ffmpeg_stderr_reader(self._ffmpeg_proc, prefix="RTP:FFmpeg")
            self.rtp_open = self._ffmpeg_proc.stdin is not None
            if self.rtp_open:
                self._backend = 'ffmpeg'
        except Exception as exc:
            self._ffmpeg_proc = None
            self.rtp_open = False
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"RTP:Failed to start FFmpeg: {exc}"))

    def _start_ffmpeg_stderr_reader(self, proc, prefix: str):
        """Drain FFmpeg stderr in a background thread to prevent pipe blockage."""

        try:
            stderr = getattr(proc, 'stderr', None)
            if stderr is None:
                return

            def _reader():
                try:
                    for raw in iter(stderr.readline, b''):
                        if not raw:
                            break
                        try:
                            line = raw.decode(errors='replace').strip()
                        except Exception:
                            line = str(raw)
                        if not line:
                            continue
                        if not self.log.full():
                            self.log.put_nowait((logging.ERROR, f"{prefix}:{line}"))
                except Exception:
                    pass

            self._ffmpeg_stderr_thread = Thread(target=_reader, daemon=True)
            self._ffmpeg_stderr_thread.start()
        except Exception:
            pass

    def _close_stream(self):
        try:
            if getattr(self, 'rtp', None) is not None:
                self.rtp.release()
        except Exception:
            pass
        self.rtp = None

        proc = getattr(self, '_ffmpeg_proc', None)
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        self._ffmpeg_proc = None

    def _write_frame(self, frame):
        """Write one frame to the active backend."""
        if frame is None:
            return

        # Ensure contiguous uint8 HxWx3
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        if self._backend == 'opencv':
            if self.rtp is not None:
                self.rtp.write(frame)
            return

        if self._backend == 'ffmpeg':
            proc = getattr(self, '_ffmpeg_proc', None)
            if proc is None or proc.stdin is None:
                return
            proc.stdin.write(frame.tobytes())
            return


if __name__ == '__main__':

    import numpy as np

    fps=  30
    display_interval = 1./fps
    height =540
    width = 720
    depth = 3

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("RTP")
   
    # Setting up RTP
    rtp = rtpServer(
        resolution =(width,height),
        fps        = fps, 
        host       = '127.0.0.1', 
        port       = 554,
        bitrate    = 2048, 
        gpu        = False)
    logger.log(logging.DEBUG, "Starting RTP Server")
    rtp.start()

    # synthetic image
    cube = np.random.randint(0, 255, (height, width, depth), 'uint8') 

    last_display = time.time()
    num_frame = 0
    
    while True:
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            num_frame += 1
            last_display = current_time

            if not rtp.queue.full():
                rtp.queue.put_nowait(cube)

        while not rtp.log.empty():
            (level, msg)=rtp.log.get_nowait()
            logger.log(level, "RTP:{}".format(msg))

    rtp.stop()
