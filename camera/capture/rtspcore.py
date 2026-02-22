###############################################################################
# RTSP video capture core
#
# Non-threaded core used by threaded and Qt wrappers.
#
# Urs Utzinger
# GPT-5.2
###############################################################################
# Changes:
# 2026 Core, qt wrapper, non-qt wrapper
# 2025 GI/appsink backend + cross-platform fallbacks
# 2021 Initial release
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: RtspCore
#
# Public methods:
# - open_cam()         : Open and configure the RTSP stream.
# - close_cam()        : Stop and close the stream.
# - capture_array()    : tuple[np.ndarray | None, float | None]
#                        Capture a frame and timestamp (ms).
# - log_stream_options(): Log the active backend / pipeline (best-effort).
# - get_control(name: str) -> Any  (currently returns metadata fields if present)
#
# Convenience properties:
# - cam_open: bool
# - buffer: FrameBuffer | None  (allocated lazily)
# - backend: str | None         ('gi' or 'opencv')
#
# Supported config parameters (configs dict):
# -------------------------------------------
# - rtsp: str                       RTSP URL
# - output_res: tuple[int,int]      Optional CPU resize target (w,h)
# - flip: int                       0..7 (same enum as cv2Capture)
# - prefer_gi: bool                 Prefer GI/GStreamer appsink when available
# - gpu: bool                       Prefer GPU decode elements on non-Linux
# - rtsp_latency: int               GStreamer rtspsrc latency (ms), default 10
# - buffersize: int                 FrameBuffer capacity (default: 32)
# - buffer_overwrite: bool          If True, FrameBuffer overwrites oldest frames when full
#
# Notes:
# - The core does not push frames into the FrameBuffer; wrappers do.
# - The FrameBuffer is allocated lazily based on the first frame (or output_res).
###############################################################################

from __future__ import annotations

from threading import Lock
from queue import Queue
import logging
import platform
import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from .framebuffer import FrameBuffer

# GStreamer (direct)
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstApp, GstVideo
except Exception:  # pragma: no cover
    Gst = None
    GstApp = None
    GstVideo = None


class RtspCore:
    """
    Non-threaded RTSP capture core.

    This core handles backend selection and frame acquisition. It is intended
    to be wrapped by threaded/Qt wrappers that push frames into FrameBuffer.
    """

    def __init__(
        self,
        configs: dict | None,
        rtsp: str | None = None,
        gpu: bool = False,
        log_queue: Queue | None = None,
    ) -> None:
        self._configs = configs or {}
        self._rtsp = rtsp if rtsp is not None else self._configs.get('rtsp', None)

        self._output_res = self._configs.get('output_res', (-1, -1))
        self._output_width = int(self._output_res[0]) if self._output_res else -1
        self._output_height = int(self._output_res[1]) if self._output_res else -1
        self._flip_method = int(self._configs.get('flip', 0) or 0)

        self._prefer_gi = bool(self._configs.get('prefer_gi', True))
        self._gpuavail = bool(self._configs.get('gpu', gpu))
        self._latency_ms = int(self._configs.get('rtsp_latency', self._configs.get('latency', 10)))

        # FrameBuffer config (used when allocating lazily)
        self._buffer_capacity = int(self._configs.get('buffersize', 32))
        if self._buffer_capacity < 1:
            self._buffer_capacity = 1
        self._buffer_overwrite = bool(self._configs.get('buffer_overwrite', True))
        self._buffer: FrameBuffer | None = None

        self._backend: str | None = None
        self._gst_pipeline_str: str | None = None
        self._priming_frame: np.ndarray | None = None
        self._priming_ts: float | None = None

        self.cam_open = False
        self.cam_lock = Lock()

        self.pipeline = None
        self.appsink = None
        self.cam = None

        self._metadata: dict | None = None

        self._log_queue = log_queue

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, level: int, msg: str) -> None:
        try:
            if self._log_queue is not None and not self._log_queue.full():
                self._log_queue.put_nowait((level, msg))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> FrameBuffer | None:
        return self._buffer

    @property
    def backend(self) -> str | None:
        return self._backend

    def get_control(self, name: str):
        if self._metadata is None:
            return None
        return self._metadata.get(name)

    # ------------------------------------------------------------------
    # Open / Close
    # ------------------------------------------------------------------

    def open_cam(self) -> bool:
        """Open the RTSP stream."""
        if not self._rtsp:
            self.cam_open = False
            self._log(logging.CRITICAL, "RTSP:Missing 'rtsp' URL in configs and no rtsp argument provided")
            return False

        self._buffer = None
        self._priming_frame = None
        self._priming_ts = None

        self.pipeline = None
        self.appsink = None
        self.cam = None
        self.cam_open = False
        self._backend = None

        opened = False

        if self._prefer_gi and Gst is not None:
            opened = self._open_gi_rtsp_pipeline()
            if opened:
                self._backend = 'gi'
                self.cam_open = True

        if not opened:
            opened = self._open_opencv_rtsp()
            if opened:
                self._backend = 'opencv'
                self.cam_open = True

        # Linux extra fallback: OpenCV + CAP_GSTREAMER pipeline
        if not opened and platform.system() == "Linux":
            opened = self._open_opencv_gst_pipeline()
            if opened:
                self._backend = 'opencv'
                self.cam_open = True

        if not self.cam_open:
            self._log(logging.CRITICAL, "RTSP:Failed to open stream")
            return False

        # If output resolution is provided, we can allocate a buffer immediately.
        if self._output_width > 0 and self._output_height > 0:
            try:
                self._buffer = FrameBuffer(
                    capacity=self._buffer_capacity,
                    frame_shape=(self._output_height, self._output_width, 3),
                    dtype=np.uint8,
                    overwrite=self._buffer_overwrite,
                )
            except Exception:
                self._buffer = None

        return True

    def close_cam(self) -> None:
        """Release the underlying capture backend (idempotent)."""
        try:
            with self.cam_lock:
                if self.cam is not None:
                    self.cam.release()
            self.cam = None
        except Exception:
            self.cam = None

        if self.pipeline is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None

        self.cam_open = False
        self._backend = None
        self._buffer = None

    # ------------------------------------------------------------------
    # Backend open helpers
    # ------------------------------------------------------------------

    def _open_opencv_rtsp(self) -> bool:
        """Open RTSP using OpenCV (typically FFmpeg backend on Windows/macOS)."""
        try:
            api = getattr(cv2, "CAP_FFMPEG", 0)
            if api:
                self.cam = cv2.VideoCapture(self._rtsp, apiPreference=api)
            else:
                self.cam = cv2.VideoCapture(self._rtsp)
            return bool(self.cam is not None and self.cam.isOpened())
        except Exception:
            self.cam = None
            return False

    def _open_opencv_gst_pipeline(self) -> bool:
        """Open RTSP using OpenCV + CAP_GSTREAMER pipeline (Linux fallback)."""
        gst = f'rtspsrc location={self._rtsp} latency={self._latency_ms} ! rtph264depay ! h264parse ! '

        if platform.machine() == 'aarch64':
            gst = gst + 'omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
        elif platform.machine() in ('armv6l', 'armv7l'):
            gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            gst = gst + 'avdec_h264 ! videoconvert ! '

        gst = gst + 'videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'
        self._gst_pipeline_str = gst
        self._log(logging.INFO, gst)

        try:
            self.cam = cv2.VideoCapture(gst, apiPreference=cv2.CAP_GSTREAMER)
            return bool(self.cam is not None and self.cam.isOpened())
        except Exception:
            self.cam = None
            return False

    def _open_gi_rtsp_pipeline(self) -> bool:
        """Open RTSP using GI GStreamer pipeline + appsink (H264)."""
        if Gst is None:
            return False

        if not hasattr(self.__class__, "_gst_inited"):
            Gst.init(None)
            self.__class__._gst_inited = True

        gst = f'rtspsrc location={self._rtsp} latency={self._latency_ms} ! rtph264depay ! h264parse ! '

        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64':
                gst = gst + 'omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
            elif platform.machine() in ('armv6l', 'armv7l'):
                gst = gst + 'v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'avdec_h264 ! videoconvert ! '

        gst = (
            gst
            + 'videoconvert ! video/x-raw,format=BGR '
            + '! queue leaky=downstream max-size-buffers=1 '
            + '! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true enable-last-sample=false qos=false'
        )

        self._gst_pipeline_str = gst
        self._log(logging.INFO, gst)

        try:
            self.pipeline = Gst.parse_launch(gst)
            self.appsink = self.pipeline.get_by_name("appsink")
            if self.appsink is None:
                raise RuntimeError("appsink element not found in pipeline")

            try:
                self.appsink.set_property("sync", False)
                self.appsink.set_property("drop", True)
                self.appsink.set_property("max-buffers", 1)
                if self.appsink.find_property("enable-last-sample") is not None:
                    self.appsink.set_property("enable-last-sample", False)
                if self.appsink.find_property("qos") is not None:
                    self.appsink.set_property("qos", False)
            except Exception:
                pass

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            return ret != Gst.StateChangeReturn.FAILURE
        except Exception as exc:
            self.pipeline = None
            self.appsink = None
            self._log(logging.WARNING, f"RTSP:GI pipeline open failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_array(self) -> tuple[np.ndarray | None, float | None]:
        """Capture a single frame and timestamp in milliseconds."""
        if not self.cam_open:
            return None, None

        if self._priming_frame is not None:
            frame = self._priming_frame
            ts_ms = self._priming_ts
            self._priming_frame = None
            self._priming_ts = None
            return frame, ts_ms

        img = None

        if self._backend == 'gi' and self.appsink is not None:
            try_pull = getattr(self.appsink, "try_pull_sample", None)
            if callable(try_pull):
                sample = try_pull(int(250 * 1e6))
            else:
                sample = self.appsink.emit("try-pull-sample", int(250 * 1e6))
            if sample is not None:
                try:
                    img = self._sample_to_bgr(sample)
                except Exception as exc:
                    img = None
                    self._log(logging.ERROR, f"RTSP:Failed to convert sample to BGR: {exc}")

        if self._backend == 'opencv' and self.cam is not None:
            try:
                with self.cam_lock:
                    ret, img_cv = self.cam.read()
                if ret and img_cv is not None:
                    img = img_cv
            except Exception:
                img = None

        if img is None:
            return None, None

        img = self._apply_resize_flip(img)
        ts_ms = time.perf_counter() * 1000.0
        return img, ts_ms

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_resize_flip(self, img: np.ndarray) -> np.ndarray:
        out = img
        if self._output_width > 0 and self._output_height > 0:
            h, w = img.shape[:2]
            if w != self._output_width or h != self._output_height:
                out = cv2.resize(out, (self._output_width, self._output_height))

        if self._flip_method != 0:
            f = self._flip_method
            if f == 1:
                out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif f == 2:
                out = cv2.rotate(out, cv2.ROTATE_180)
            elif f == 3:
                out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            elif f == 4:
                out = cv2.flip(out, 1)
            elif f == 5:
                out = cv2.flip(cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
            elif f == 6:
                out = cv2.flip(out, 0)
            elif f == 7:
                out = cv2.transpose(out)
        return out

    def _sample_to_bgr(self, sample):
        """Convert a Gst.Sample (video/x-raw,format=BGR) into a NumPy array."""
        if sample is None or GstVideo is None:
            return None

        caps = sample.get_caps()
        if caps is None:
            return None

        info_new = getattr(GstVideo.VideoInfo, "new_from_caps", None)
        if callable(info_new):
            info = info_new(caps)
            if info is None:
                return None
        else:
            info = GstVideo.VideoInfo()
            ok = info.from_caps(caps)
            if not ok:
                return None

        buf = sample.get_buffer()
        if buf is None:
            return None

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            height = int(info.height)
            width = int(info.width)
            stride = int(info.stride[0]) if info.stride and info.stride[0] else width * 3

            data = np.frombuffer(map_info.data, dtype=np.uint8)
            if data.size < height * stride:
                return None

            frame_2d = data[: height * stride].reshape((height, stride))
            frame = frame_2d[:, : width * 3].reshape((height, width, 3))
            return frame.copy()
        finally:
            buf.unmap(map_info)

    def log_stream_options(self) -> None:
        if self._backend == 'gi' and self._gst_pipeline_str:
            self._log(logging.INFO, self._gst_pipeline_str)
        elif self._backend == 'opencv':
            self._log(logging.INFO, f"RTSP:OpenCV backend url={self._rtsp}")

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_output_res(self, res: tuple[int, int]) -> None:
        try:
            w, h = int(res[0]), int(res[1])
        except Exception:
            return
        self._output_res = (w, h)
        self._output_width = w
        self._output_height = h

    def set_flip(self, flip: int) -> None:
        try:
            self._flip_method = int(flip)
        except Exception:
            pass
