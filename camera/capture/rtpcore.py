###############################################################################
# RTP (UDP) video capture core
#
# Non-threaded core used by threaded and Qt wrappers.
#
# Urs Utzinger
# GPT-5.2
###############################################################################
# Changes:
# 2026 Core, qt wrapper, non-qt wrapper
# 2025 GI/appsink backend + FFmpeg fallback
# 2021 Initial release
###############################################################################

###############################################################################
# Public API & Supported Config
#
# Class: RtpCore
#
# Public methods:
# - open_cam():                   Open and configure the RTP stream.
# - close_cam():                  Stop and close the stream.
# - capture_array(): tuple[np.ndarray | None, float | None]
#                                 Capture a frame and timestamp (ms).
# - log_stream_options():         Log the active backend / pipeline (best-effort).
# - get_control(name: str) -> Any (currently returns metadata fields if present)
#
# Convenience properties:
# - cam_open: bool
# - buffer: FrameBuffer | None    (allocated lazily)
# - backend: str | None           ('gi' | 'opencv' | 'ffmpeg')
#
# Supported config parameters (configs dict):
# -------------------------------------------
# - port: int                     UDP port for RTP stream (default: 554)
# - output_res: tuple[int,int]    Optional CPU resize target (w,h)
# - camera_res: tuple[int,int]    Expected source size (used by FFmpeg fallback)
# - flip: int                     0..7 (same enum as cv2Capture)
# - prefer_gi: bool               Prefer GI/GStreamer appsink when available
# - gpu: bool                     Prefer GPU decode elements on non-Linux
# - rtp_encoding_name: str        RTP encoding name (default: 'H264')
# - rtp_payload: int              RTP payload type (default: 96)
# - rtp_clock_rate: int           RTP clock rate (default: 90000)
# - rtp_sdp / sdp: str            SDP file for FFmpeg fallback
# - buffersize: int               FrameBuffer capacity (default: 32)
# - buffer_overwrite: bool        If True, FrameBuffer overwrites oldest frames when full
#
# Notes:
# - The core does not push frames into the FrameBuffer; wrappers do.
# - The FrameBuffer is allocated lazily based on the first frame (or output_res).
###############################################################################

from __future__ import annotations

from threading import Lock, Thread
from queue import Queue
import logging
import platform
import subprocess
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

# FFmpeg helper (bundled binary via pip)
try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


class RtpCore:
    """
    Non-threaded RTP capture core.

    This core handles backend selection and frame acquisition. It is intended
    to be wrapped by threaded/Qt wrappers that push frames into FrameBuffer.
    """

    def __init__(
        self,
        configs: dict | None,
        port: int | None = None,
        gpu: bool = False,
        log_queue: Queue | None = None,
    ) -> None:
        self._configs = configs or {}

        self._port = int(port) if port is not None else int(self._configs.get('port', 554))
        self._output_res = self._configs.get('output_res', (-1, -1))
        self._output_width = int(self._output_res[0]) if self._output_res else -1
        self._output_height = int(self._output_res[1]) if self._output_res else -1
        self._flip_method = int(self._configs.get('flip', 0) or 0)

        self._prefer_gi = bool(self._configs.get('prefer_gi', True))
        self._gpuavail = bool(self._configs.get('gpu', gpu))

        self._rtp_encoding_name = self._configs.get('rtp_encoding_name', 'H264')
        self._rtp_payload = int(self._configs.get('rtp_payload', 96))
        self._rtp_clock_rate = int(self._configs.get('rtp_clock_rate', 90000))

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

        # FFmpeg fallback
        self._ffmpeg_proc = None
        self._ffmpeg_width = None
        self._ffmpeg_height = None
        self._ffmpeg_frame_bytes = None

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
        """Open the RTP stream."""
        self._buffer = None
        self._priming_frame = None
        self._priming_ts = None

        self.pipeline = None
        self.appsink = None
        self.cam = None
        self.cam_open = False
        self._backend = None

        # Build default RTP pipeline
        gst = (
            'udpsrc port={:d} caps=application/x-rtp,media=(string)video,clock-rate=(int){:d},'
            'encoding-name=(string){:s},payload=(int){:d} ! '
        ).format(self._port, self._rtp_clock_rate, self._rtp_encoding_name, self._rtp_payload)

        if str(self._rtp_encoding_name).upper() == 'H264':
            gst = gst + 'rtph264depay ! '
        else:
            self._log(logging.CRITICAL, f"RTP:Unsupported rtp_encoding_name '{self._rtp_encoding_name}'. Only H264 is built-in.")
            self.cam_open = False
            return False

        plat = platform.system()
        if plat == "Linux":
            if platform.machine() == 'aarch64':
                gst = gst + 'h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! '
            elif platform.machine() in ('armv6l', 'armv7l'):
                gst = gst + 'h264parse ! v4l2h264dec capture-io-mode=4 ! v4l2convert output-io-mode=5 capture-io-mode=4 ! '
        else:
            if self._gpuavail:
                gst = gst + 'nvh264dec ! videoconvert ! '
            else:
                gst = gst + 'decodebin ! videoconvert ! '

        gst = (
            gst
            + 'videoconvert ! video/x-raw,format=BGR '
            + '! queue leaky=downstream max-size-buffers=1 '
            + '! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true enable-last-sample=false qos=false'
        )

        self._gst_pipeline_str = gst
        self._log(logging.INFO, gst)

        # Prefer GI-based pipeline
        if self._prefer_gi and Gst is not None:
            if not hasattr(self.__class__, "_gst_inited"):
                Gst.init(None)
                self.__class__._gst_inited = True
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
                self.cam_open = ret != Gst.StateChangeReturn.FAILURE
                if self.cam_open:
                    self._backend = 'gi'
            except Exception as exc:
                self.pipeline = None
                self.appsink = None
                self.cam_open = False
                self._log(logging.CRITICAL, f"RTP:Failed to open GI GStreamer pipeline: {exc}")

        # Fallback 1: OpenCV VideoCapture (CAP_GSTREAMER)
        if not self.cam_open:
            try:
                self.cam = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                self.cam_open = self.cam.isOpened()
                if self.cam_open:
                    self._backend = 'opencv'
            except Exception:
                self.cam = None
                self.cam_open = False

        # Fallback 2: FFmpeg subprocess
        if not self.cam_open:
            opened = self._open_ffmpeg_rtp()
            if opened:
                self.cam_open = True
                self._backend = 'ffmpeg'

        if not self.cam_open:
            self._log(logging.CRITICAL, "RTP:Failed to open stream")
            return False

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
            if self.cam is not None:
                self.cam.release()
        except Exception:
            pass
        self.cam = None

        if self.pipeline is not None and Gst is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsink = None

        proc = self._ffmpeg_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        self._ffmpeg_proc = None

        self.cam_open = False
        self._backend = None
        self._buffer = None

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
                    self._log(logging.ERROR, f"RTP:Failed to convert sample to BGR: {exc}")

        if self._backend == 'opencv' and self.cam is not None:
            try:
                with self.cam_lock:
                    ret, img_cv = self.cam.read()
                if ret and img_cv is not None:
                    img = img_cv
            except Exception:
                img = None

        if self._backend == 'ffmpeg':
            img = self._ffmpeg_read_frame()

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

    def _open_ffmpeg_rtp(self) -> bool:
        """Open RTP via ffmpeg subprocess (cross-platform fallback)."""
        if imageio_ffmpeg is None:
            return False

        sdp_path = self._configs.get('rtp_sdp', self._configs.get('sdp', None))
        if not sdp_path:
            self._log(logging.CRITICAL, "RTP:FFmpeg fallback requires configs['rtp_sdp'] (SDP file path)")
            return False

        out_res = self._configs.get('output_res', (-1, -1))
        cam_res = self._configs.get('camera_res', (-1, -1))
        if out_res[0] > 0 and out_res[1] > 0:
            width, height = int(out_res[0]), int(out_res[1])
        elif cam_res[0] > 0 and cam_res[1] > 0:
            width, height = int(cam_res[0]), int(cam_res[1])
        else:
            self._log(logging.CRITICAL, "RTP:FFmpeg fallback needs configs['camera_res'] or configs['output_res']")
            return False

        self._ffmpeg_width = width
        self._ffmpeg_height = height
        self._ffmpeg_frame_bytes = width * height * 3

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            '-loglevel', 'error',
            '-hide_banner',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-analyzeduration', '0',
            '-probesize', '32',
            '-protocol_whitelist', 'file,udp,rtp',
            '-i', str(sdp_path),
            '-an',
            '-vf', f'scale={width}:{height}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            'pipe:1',
        ]

        self._log(logging.INFO, f"RTP:FFmpeg cmd: {' '.join(cmd)}")

        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._start_ffmpeg_stderr_reader(self._ffmpeg_proc, prefix="RTP:FFmpeg")
            return self._ffmpeg_proc.stdout is not None
        except Exception as exc:
            self._ffmpeg_proc = None
            self._log(logging.CRITICAL, f"RTP:Failed to start FFmpeg: {exc}")
            return False

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
                        self._log(logging.ERROR, f"{prefix}:{line}")
                except Exception:
                    pass

            t = Thread(target=_reader, daemon=True)
            t.start()
        except Exception:
            pass

    def _ffmpeg_read_frame(self):
        proc = self._ffmpeg_proc
        if proc is None or proc.stdout is None:
            return None

        n = self._ffmpeg_frame_bytes
        if not n:
            return None

        try:
            raw = proc.stdout.read(n)
        except Exception:
            return None

        if raw is None or len(raw) != n:
            return None

        h = int(self._ffmpeg_height)
        w = int(self._ffmpeg_width)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        return frame

    def log_stream_options(self) -> None:
        if self._backend == 'gi' and self._gst_pipeline_str:
            self._log(logging.INFO, self._gst_pipeline_str)
        elif self._backend == 'opencv':
            self._log(logging.INFO, f"RTP:OpenCV backend port={self._port}")
        elif self._backend == 'ffmpeg':
            self._log(logging.INFO, "RTP:FFmpeg backend active")

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
