###############################################################################
# RTSP server
#
# Cross-platform strategy:
# - Preferred backend (when available): GStreamer RTSPServer via PyGObject (gi).
# - Fallback backend (pip-friendly): FFmpeg listen-mode RTSP server via `imageio-ffmpeg`.
#
# This mirrors the capture-side approach in `rtspCapture` (GI/appsink preferred,
# otherwise OpenCV+FFmpeg).
# 
# Create Stream with 
# rtsp = rtspServer(
#     resolution =(width,height),
#     fps        = fps,
#     bitrate    = 2048,
#     stream_uri = '/test')
#
# rtsp.start()
# cube = np.random.randint(0, 255, (height, width, depth), 'uint8')
# rtsp.queue.put_nowait(cube)
#
# Urs Utzinger 
#
# Changes:
# 2025 Codereview, switched to GI and Fallback FFmpeg added
# 2022 Initial Release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Event
from queue import Queue
from queue import Empty

# System
import logging
import time
import platform
import subprocess
import os

# GStreamer RTSPServer (preferred when GI is available)
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GstRtspServer, GLib
except Exception:  # pragma: no cover
    Gst = None
    GstRtspServer = None
    GLib = None

# FFmpeg helper (bundled binary via pip)
try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None

###############################################################################
# RTSP Server
###############################################################################

if GstRtspServer is not None:

    class rtspFactory(GstRtspServer.RTSPMediaFactory):
        _STOP_ITEM = (None, None)

        def __init__(
            self,
            resolution: tuple[int, int],
            fps: int,
            bitrate: int,
            queue: Queue,
            log: Queue,
            stop_event: Event,
            **args,
        ):
            super(rtspFactory, self).__init__(**args)

            self.queue = queue
            self.log = log
            self._stop_event = stop_event

            self._res = resolution
            self._fps = int(fps)
            self._bitrate = int(bitrate)

            self._number_frames = 0
            self._duration = int(Gst.SECOND / max(self._fps, 1))

            s_src  = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                     'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 '.format(self._res[0], self._res[1], self._fps)

            s_h264 = '! videoconvert ! video/x-raw,format=I420 ' \
                 '! x264enc speed-preset=ultrafast tune=zerolatency bitrate={} threads=4 '.format(max(self._bitrate // 1000, 1))

            self.pipeline_str = "( {}! queue max-size-buffers=1 name=q_enc {}! rtph264pay config-interval=1 name=pay0 pt=96 )".format(s_src, s_h264)
            if not self.log.full():
                self.log.put_nowait((logging.DEBUG, "RTSP:create stream {}".format(self.pipeline_str)))
        
        def on_need_data(self, src, length):
            if self._stop_event.is_set():
                return

            try:
                item = self.queue.get(timeout=0.25)
            except Empty:
                return

            if item == self._STOP_ITEM:
                return

            frame_time = None
            frame = None
            if isinstance(item, tuple) and len(item) == 2:
                frame_time, frame = item
            else:
                frame = item

            if frame is None:
                return

            try:
                data = frame.tobytes()
            except Exception as exc:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"RTSP:frame conversion failed: {exc}"))
                return

            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self._duration

            if isinstance(frame_time, (int, float)):
                timestamp = int(frame_time * Gst.SECOND)
            else:
                timestamp = int(self._number_frames * self._duration)

            buf.pts = buf.dts = timestamp
            buf.offset = timestamp
            self._number_frames += 1

            retval = src.emit('push-buffer', buf)
            if retval != Gst.FlowReturn.OK:
                if not self.log.full():
                    self.log.put_nowait((logging.DEBUG, "RTSP:pushed buffer error {}".format(retval)))

        def do_create_element(self, url):
            return Gst.parse_launch(self.pipeline_str)


        def do_configure(self, rtsp_media):
            self._number_frames = 0
            appsrc = rtsp_media.get_element().get_child_by_name('source')
            appsrc.connect('need-data', self.on_need_data)
            
class rtspServer(Thread):
            
    def __init__(
        self,
        resolution: tuple = (320, 240),
        fps: int = 15,
        bitrate: int = 2048,
        stream_uri: str = '/test',
        host: str = '0.0.0.0',
        port: int = 8554,
        prefer_gi: bool = True,
        queue_size: int = 32,
        **args,
    ):

        super().__init__(daemon=True)

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self._stop_event     = Event()

        # populate desired settings from configuration file or function call
        self._res     = resolution
        self._fps     = int(fps)
        self._bitrate = int(bitrate)
        self._uri     = stream_uri
        self._host    = host
        self._port    = int(port)
        self._prefer_gi = bool(prefer_gi)

        self._backend = None
        self._loop = None
        self._server = None
        self.factory = None
        self._ffmpeg_proc = None
        self._ffmpeg_stderr_thread = None

        self._open_stream(**args)
    
    def stop(self):
        self._stop_event.set()
        try:
            if not self.queue.full():
                self.queue.put_nowait((None, None))
        except Exception:
            pass

        if self._backend == 'gi':
            loop = getattr(self, '_loop', None)
            if loop is not None and GLib is not None:
                try:
                    GLib.idle_add(loop.quit)
                except Exception:
                    try:
                        loop.quit()
                    except Exception:
                        pass

        if self._backend == 'ffmpeg':
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

    def close(self):
        self.stop()

    def run(self):
        if self._backend == 'gi':
            # Attach server to default main context in the thread that runs the loop
            self._server.attach(None)
            self._loop = GLib.MainLoop()
            try:
                self._loop.run()
            finally:
                self._loop = None
            return

        if self._backend == 'ffmpeg':
            self._ffmpeg_update_loop()

    def _open_stream(self, **args):
        """Choose backend once at open time.

        Preferred: GI GStreamer RTSPServer when available.
        Fallback: FFmpeg listen-mode RTSP server (imageio-ffmpeg).
        """

        self._backend = None

        if self._prefer_gi and (GstRtspServer is not None) and (Gst is not None) and (GLib is not None):
            try:
                Gst.init(None)
                self._server = GstRtspServer.RTSPServer(**args)
                # Bind address/port if supported
                try:
                    self._server.set_service(str(self._port))
                except Exception:
                    pass
                try:
                    self._server.set_address(str(self._host))
                except Exception:
                    try:
                        self._server.set_property('address', str(self._host))
                    except Exception:
                        pass

                self.factory = rtspFactory(
                    resolution=self._res,
                    fps=self._fps,
                    bitrate=self._bitrate,
                    queue=self.queue,
                    log=self.log,
                    stop_event=self._stop_event,
                )
                self.factory.set_shared(True)

                self._mountpoints = self._server.get_mount_points()
                self._mountpoints.add_factory(self._uri, self.factory)

                self._backend = 'gi'
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"RTSP:Backend: GI/GStreamer (rtsp://{self._host}:{self._port}{self._uri})"))
                return
            except Exception as exc:
                self._server = None
                self.factory = None
                self._backend = None
                if not self.log.full():
                    self.log.put_nowait((logging.WARNING, f"RTSP:GI backend open failed: {exc}"))

        # FFmpeg fallback
        if imageio_ffmpeg is None:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "RTSP:No GI/GStreamer and imageio-ffmpeg not installed"))
            return

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        width, height = int(self._res[0]), int(self._res[1])
        uri = self._uri
        if not uri.startswith('/'):
            uri = '/' + uri

        # ffmpeg acts as RTSP server when using -rtsp_flags listen
        out_url = f"rtsp://{self._host}:{self._port}{uri}"

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
            '-f', 'rtsp',
            '-rtsp_flags', 'listen',
            out_url,
        ]

        if not self.log.full():
            self.log.put_nowait((logging.INFO, f"RTSP:Backend: FFmpeg listen-mode ({out_url})"))
            self.log.put_nowait((logging.INFO, f"RTSP:FFmpeg cmd: {' '.join(cmd)}"))

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._start_ffmpeg_stderr_reader(self._ffmpeg_proc, prefix='RTSP:FFmpeg')
            if self._ffmpeg_proc.stdin is not None:
                self._backend = 'ffmpeg'
        except Exception as exc:
            self._ffmpeg_proc = None
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"RTSP:Failed to start FFmpeg: {exc}"))

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

    def _ffmpeg_update_loop(self):
        proc = getattr(self, '_ffmpeg_proc', None)
        if proc is None or proc.stdin is None:
            return

        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.25)
            except Empty:
                item = None

            if item is None:
                continue
            if item == (None, None):
                break

            frame = None
            if isinstance(item, tuple) and len(item) == 2:
                _, frame = item
            else:
                frame = item

            if frame is None:
                continue

            try:
                data = frame.tobytes()
                proc.stdin.write(data)
            except Exception as exc:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"RTSP:FFmpeg write failed: {exc}"))
                break
   
if __name__ == '__main__':

    import numpy as np

    fps=  30
    display_interval = 1./fps
    height =480
    width = 640
    depth = 3

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("RTSP")
    
    rtsp = rtspServer(
        resolution =(width,height),
        fps        = fps,
        bitrate    = 2048,
        stream_uri = '/test')

    logger.log(logging.DEBUG, "Starting RTSP Server")
    rtsp.start()

    last_display = time.time()
    num_frame = 0
    
    stop = False
    cv2_ok = False
    try:
        import cv2
        cv2_ok = True
        cv2.namedWindow("RTSP", cv2.WINDOW_NORMAL)
    except Exception:
        cv2_ok = False

    while not stop:
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            num_frame += 1
            last_display = current_time
            # synthetic image
            cube = np.random.randint(0, 255, (height, width, depth), 'uint8') 
            if not rtsp.queue.full():
                rtsp.queue.put_nowait(cube)

        if cv2_ok:
            try:
                cv2.imshow("RTSP", cube)
                key = cv2.waitKey(1)
                if (key == 27) or (key & 0xFF == ord('q')):
                    stop = True
                if cv2.getWindowProperty("RTSP", 0) < 0:
                    stop = True
            except Exception:
                stop = True

        while not rtsp.log.empty():
            (level, msg)=rtsp.log.get_nowait()
            logger.log(level, "RTSP:{}".format(msg))

    rtsp.stop()
    rtsp.join(timeout=2.0)

    if cv2_ok:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
