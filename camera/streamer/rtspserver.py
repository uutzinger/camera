###############################################################################
# RTSP server (UNIX only)
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
# 2025 Codereview and cleanup
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

# import required librarIES like Gstreamer and GstreamerRtspServer
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

###############################################################################
# RTSP Server
###############################################################################

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
        # could remove all options provided to appsrc
        
        s_h264 = '! videoconvert ! video/x-raw,format=I420 ' \
             '! x264enc speed-preset=ultrafast tune=zerolatency bitrate={} threads=4 '.format(max(self._bitrate // 1000, 1))
        # could remove ! video/x-raw,format=I420
                
        #s_h264 = "x264enc speed-preset=ultrafast tune=zerolatency threads=4 "
        #s_src = "v4l2src ! video/x-raw,rate=30,width=320,height=240 ! videoconvert ! video/x-raw,format=I420"
        #s_h264 = "videoconvert ! vaapiencode_h264 bitrate=1000"
		#s_src  = "videotestsrc ! video/x-raw,rate=30,width=320,height=240,format=I420 "
        #s_h264 = "omx264enc control-rate=1 target-bitrate={} ".format(self._bitrate)
  
        # Make sure simple test pipelines work
        # check the audio version and add...
        # ./test-launch "( videotestsrc ! x264enc ! rtph264pay name=pay0 pt=96 )"
        # ./test-launch "( videotestsrc ! x264enc speed-preset=ultrafast ! rtph264pay name=pay0 pt=96 )"
        # ./test-launch "( videotestsrc ! x264enc speed-preset=ultrafast ! rtph264pay name=pay0 pt=96 )"
        # ./test-launch "( videotestsrc ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay name=pay0 pt=96 )"
        # ./test-launch "( videotestsrc ! x264enc speed-preset=ultrafast tune=zerolatency threads=4 ! rtph264pay name=pay0 pt=96 )"
        # ./test-launch "( videotestsrc ! x264enc speed-preset=ultrafast tune=zerolatency bitrate=1000000 threads=4 ! rtph264pay name=pay0 pt=96 )"
        # width audio
        # ./test-launch "( videotestsrc ! video/x-raw,width=352,height=288,framerate=15/1,format=I420 ! x264enc speed-preset=ultrafast tune=zerolatency bitrate=1000000 threads=4 ! rtph264pay config-interval=1 name=pay0 pt=96 audiotestsrc ! audio/x-raw,rate=8000 ! alawenc ! rtppcmapay name=pay1 pt=8 )"
        # with real video
        # ./test-launch "( v4l2src device=/dev/video0 ! video/x-h264, width=1280, height=720, framerate=15/1 ! h264parse config-interval=1 ! rtph264pay name=pay0 pt=96 )"
        # ./test-launch "( v4l2src device=/dev/video0 ! video/x-h264, width=1280, height=720, framerate=15/1 ! queue max-size-buffers=1 name=q_enc ! h264parse config-interval=1 ! rtph264pay name=pay0 pt=96 )"

        self.pipeline_str = "( {}! queue max-size-buffers=1 name=q_enc {}! rtph264pay config-interval=1 name=pay0 pt=96 )".format(s_src, s_h264)
        # coult remove queue max ... but need rtph264pay0
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
        # self.appsrc       = self.pipeline.get_child_by_index(4)
        # GstRtspServer.RTSPMediaFactory.__init__(self)


    def do_configure(self, rtsp_media):
        self._number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)
            
class rtspServer(Thread):
            
    def __init__(self, 
        resolution: tuple = (320, 240), 
        fps: int          = 15,
        bitrate: int      = 2048,
        stream_uri : str  = '/test', **args):

        super().__init__(daemon=True)

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self._stop_event     = Event()

        # populate desired settings from configuration file or function call
        self._res     = resolution
        self._fps     = fps
        self._bitrate = bitrate
        self._uri     = stream_uri    
        self._duration = 1. / self._fps * Gst.SECOND  # duration of a frame in nanoseconds

        # Gst init is cheap and idempotent
        Gst.init(None)

        self._server = GstRtspServer.RTSPServer(**args)
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
        self._loop = None
    
    def stop(self):
        self._stop_event.set()
        try:
            if not self.queue.full():
                self.queue.put_nowait(self.factory._STOP_ITEM)
        except Exception:
            pass

        loop = getattr(self, '_loop', None)
        if loop is not None:
            try:
                GLib.idle_add(loop.quit)
            except Exception:
                try:
                    loop.quit()
                except Exception:
                    pass

    def close(self):
        self.stop()

    def run(self):
        # Attach server to default main context in the thread that runs the loop
        self._server.attach(None)
        self._loop = GLib.MainLoop()
        try:
            self._loop.run()
        finally:
            self._loop = None
   
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
