###############################################################################
# RTSP server
# 2022 Initial Release
# Urs Utzinger 
###############################################################################

# Implementing GstRtspServer
# https://github.com/mad4ms/python-opencv-gstreamer-examples/blob/master/gst_rtsp_server.py#L43
# https://github.com/tamaggo/gstreamer-examples/blob/master/test_gst_rtsp_server.py
# https://github.com/BreeeZe/rpos/blob/master/python/gst-rtsp-launch.py
# https://jonic.cn/?qa=560930/
# https://www.semicolonworld.com/question/62502/how-to-convert-a-video-on-disk-to-a-rtsp-stream

# Developer solution in C
# https://github.com/GStreamer/gst-rtsp-server/blob/master/examples/test-appsrc.c

# Implement the response to this question
# https://stackoverflow.com/questions/47396372/write-opencv-frames-into-gstreamer-rtsp-server-pipeline

# Dealing with child and parent class, why we need super() 
# https://stackoverflow.com/questions/55032685/child-class-inherit-parent-class-self-variable
  
###############################################################################
# Imports
###############################################################################

# Multi Threading
from queue import Queue

# System
import logging, time
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

    def __init__(self, **args):
        super(rtspFactory, self).__init__(**args) # this should allow rtspFactory inherit self
        self._number_frames = 0
        self._duration = 1. / self._fps * Gst.SECOND
        # 
        s_src  = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                 'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 '.format(self._res[0], self.__res[1], self._fps)
        # could remove all options provided to appsrc
        
        s_h264 = '! videoconvert ! video/x-raw,format=I420 ' \
                 '! x264enc speed-preset=ultrafast tune=zerolatency bitrate={} threads=4 '.format(self._bitrate/1000)
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
        if not self.log.full(): self.log.put_nowait((logging.DEBUG, "create stream {}".format(self.pipeline_str)))
        
    def on_need_data(self, src, length):        
        if not self.stopped:
            (frame_time, frame) = self.queue.get(block=True, timeout=None)            
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self._duration
            timestamp = self._number_frames * self._duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self._number_frames += 1
            retval = src.emit('push-buffer', buf)
            if retval != Gst.FlowReturn.OK:
                if not self.log.full(): self.log.put_nowait(( logging.DEBUG, "pushed buffer error {}".format(retval) ))
            if not self.log.full(): self.log.put_nowait((logging.DEBUG, "pushed buffer,frame{},duration{}ns,durations{}s".format(self.number_frames,self.duration,self.duration / Gst.SECOND)))

    def do_create_element(self, url):
        return Gst.parse_launch(self.pipeline_str)
        # self.appsrc       = self.pipeline.get_child_by_index(4)
        # GstRtspServer.RTSPMediaFactory.__init__(self)


    def do_configure(self, rtsp_media):
        self._number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)
            
class rtspServer(GstRtspServer.RTSPServer):
            
    def __init__(self, 
        resolution: tuple = (320, 240), 
        fps: int          = 15,
        bitrate: int      = 2048,
        stream_uri : str  = '/test', **args):
        
        super(rtspServer, self).__init__(**args) # not sure what this is for as we dont need to inherit from main

        # Threading Queue, Locks, Events
        self.queue           = Queue(maxsize=32)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # populate desired settings from configuration file or function call
        self._res     = resolution
        self._fps     = fps
        self._bitrate = bitrate
        self._uri     = stream_uri    
        self._duration = 1. / self._fps * Gst.SECOND  # duration of a frame in nanoseconds
        
        # self.server     = GstRtspServer.RTSPServer()
        self.factory    = rtspFactory()
        self.factory.set_shared(True)
        self.mountpoints = self.get_mount_points()
        self.mountpoints.add_factory(self._uri, self.factory)
        self.attach(None)
        # self.server.attach(None)
        self.stopped = False
    
    def stop(self):
        self.stopped = True
        
    def start(self):
        # GObject.threads_init(), depreicated?
        Gst.init(None)
        
        self.loop = GLib.MainLoop()
        self.loop.run()
   
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
    
    while True:
        current_time = time.time()

        if (current_time - last_display) > display_interval:
            num_frame += 1
            last_display = current_time
            # synthetic image
            cube = np.random.randint(0, 255, (height, width, depth), 'uint8') 
            if not rtsp.queue.full: rtsp.queue.put_nowait(cube)

        while not rtsp.log.empty():
            (level, msg)=rtsp.log.get_nowait()
            logger.log(level, "RTSP:{}".format(msg))

    rtsp.stop()
