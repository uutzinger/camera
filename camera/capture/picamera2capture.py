###############################################################################
# Raspberry Pi CSI video capture (Picamera2 / libcamera)
#
# Uses the picamera2 library to interface with the Raspberry Pi Camera Module
#
# Mirrors the behavior of piCapture (legacy picamera) where practical:
# - Threaded capture into a bounded Queue
# - Same flip enum as cv2Capture (0..7)
# - Optional output resizing
#
# Urs Utzinger
# ChatGPT, OpenAI
#
# Changes:
# 2025 Initial Release
###############################################################################

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

from threading import Thread, Lock
from queue import Queue
import logging
import time

import cv2

try:
    from picamera2 import Picamera2
except Exception:  # pragma: no cover
    Picamera2 = None  # type: ignore


class piCamera2Capture(Thread):
    """Threaded capture for Raspberry Pi CSI cameras using Picamera2."""

    def __init__(self,
        configs: dict,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        queue_size: int = 32,
    ):
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}

        self.camera_num = camera_num

        if exposure is not None:
            self._exposure = exposure
        else:
            self._exposure = self._configs.get("exposure", -1)

        if res is not None:
            self._camera_res = res
        else:
            self._camera_res = self._configs.get("camera_res", (640, 480))

        self._capture_width = int(self._camera_res[0])
        self._capture_height = int(self._camera_res[1])

        self._output_res    = self._configs.get("output_res", (-1, -1))
        self._output_width  = int(self._output_res[0])
        self._output_height = int(self._output_res[1])

        self._framerate    = self._configs.get("fps", 30)
        self._flip_method  = self._configs.get("flip", 0)
        self._autoexposure = self._configs.get("autoexposure", -1)
        self._autowb       = self._configs.get("autowb", -1)
        self._fourcc       = str(self._configs.get("fourcc", "")).upper()
        # Control whether to use libcamera hardware transform
        self._hw_transform = bool(self._configs.get("hw_transform", 1))
        # Resolved format and transform flags
        self._format: str | None = None
        self._transform_applied: bool = False

        # Threading
        # Allow override via configs['buffersize']
        queue_size = int(self._configs.get("buffersize", queue_size))
        self.capture = Queue(maxsize=queue_size)
        self.log = Queue(maxsize=32)
        self.stopped = True
        self.cam_lock = Lock()

        # Runtime
        self.picam2 = None
        self.cam_open = False
        self._last_metadata: dict | None = None
        self._camera_controls: dict | None = None
        self.frame_time = 0.0
        self.measured_fps = 0.0

        self.open_cam()

    def stop(self):
        """Stop the thread."""
        self.stopped = True

    def start(self):
        """Start the capture thread."""
        self.stopped = False
        super().start()

    def run(self):
        """Thread entrypoint."""
        if not self.cam_open:
            return

        last_time = time.perf_counter()
        num_frames = 0

        try:
            while not self.stopped:
                current_time = time.perf_counter()

                if self.picam2 is None:
                    time.sleep(0.01)
                    continue

                try:
                    with self.cam_lock:
                        img = self.picam2.capture_array("main")
                        self._last_metadata = self._capture_metadata_nolock()
                except Exception as exc:
                    if not self.log.full():
                        self.log.put_nowait((logging.WARNING, f"PiCam2:Capture failed: {exc}"))
                    time.sleep(0.005)
                    continue

                if img is None:
                    time.sleep(0.005)
                    continue

                num_frames += 1
                self.frame_time = int(current_time * 1000)

                img_proc = img
                is_yuv420 = (self._format == "YUV420")

                # If the configured format yields 4 channels, drop alpha.
                try:
                    if (not is_yuv420) and getattr(img_proc, "ndim", 0) == 3 and img_proc.shape[2] == 4:
                        img_proc = img_proc[:, :, :3]
                except Exception:
                    pass

                # Resize only if an explicit output size was provided
                if (self._output_width > 0) and (self._output_height > 0) and (not is_yuv420):
                    img_proc = cv2.resize(img_proc, self._output_res)

                # Apply flip/rotation if requested (same enum as cv2Capture)
                if (self._flip_method != 0) and (not self._transform_applied) and (not is_yuv420):
                    if self._flip_method == 1:  # ccw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 2:  # rot 180
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_180)
                    elif self._flip_method == 3:  # cw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_CLOCKWISE)
                    elif self._flip_method == 4:  # horizontal (left-right)
                        img_proc = cv2.flip(img_proc, 1)
                    elif self._flip_method == 5:  # upright diagonal. ccw & lr
                        img_proc = cv2.flip(cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                    elif self._flip_method == 6:  # vertical (up-down)
                        img_proc = cv2.flip(img_proc, 0)
                    elif self._flip_method == 7:  # upperleft diagonal
                        img_proc = cv2.transpose(img_proc)

                if not self.capture.full():
                    self.capture.put_nowait((self.frame_time, img_proc))
                else:
                    if not self.log.full():
                        self.log.put_nowait((logging.WARNING, "PiCam2:Capture Queue is full!"))

                # FPS calculation
                if (current_time - last_time) >= 5.0:
                    self.measured_fps = num_frames / 5.0
                    if not self.log.full():
                        self.log.put_nowait((logging.INFO, f"PiCam2:FPS:{self.measured_fps}"))
                    last_time = current_time
                    num_frames = 0
        finally:
            self.close_cam()

    def open_cam(self):
        if Picamera2 is None:
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, "PiCam2:picamera2 is not installed"))
            self.cam_open = False
            return

        try:
            self.picam2 = Picamera2(camera_num=self.camera_num)
            try:
                self._camera_controls = getattr(self.picam2, "camera_controls", None)
            except Exception:
                self._camera_controls = None

            # Prefer BGR888 so frames are OpenCV-ready unless user requested YUV.
            picam_format = self._map_fourcc(self._fourcc) or "BGR888"
            self._format = picam_format
            
            # Choose main stream size: honor output_res when provided (avoids CPU resize)
            if (self._output_width > 0) and (self._output_height > 0):
                main_size = (self._output_width, self._output_height)
            else:
                main_size = (self._capture_width, self._capture_height)

            # Try hardware transform via libcamera Transform (only if needed)
            transform = None
            if self._hw_transform and (self._flip_method != 0):
                try:
                    from libcamera import Transform  # type: ignore
                    transform = self._flip_to_transform(self._flip_method, Transform)
                except Exception:
                    transform = None
            self._transform_applied = transform is not None and self._flip_method != 0
            config = None
            try:
                kwargs = dict(main={"size": main_size, "format": picam_format},
                              controls={"FrameRate": self._framerate})
                if transform is not None:
                    kwargs["transform"] = transform
                config = self.picam2.create_video_configuration(**kwargs)
            except Exception:
                # Fallbacks
                for fmt in ("BGR888", "RGB888"):
                    try:
                        kwargs = dict(main={"size": main_size, "format": fmt},
                                      controls={"FrameRate": self._framerate})
                        if transform is not None:
                            kwargs["transform"] = transform
                        config = self.picam2.create_video_configuration(**kwargs)
                        self._format = fmt
                        break
                    except Exception:
                        continue

            self.picam2.configure(config)

            # Apply initial controls
            controls = {}

            exposure = self._exposure
            autoexp  = self._autoexposure
            autowb   = self._autowb

            manual_requested = exposure is not None and exposure > 0
            if manual_requested:
                controls["AeEnable"] = False
                controls["ExposureTime"] = int(exposure)
            else:
                if autoexp is None or autoexp == -1:
                    pass
                elif autoexp > 0:
                    controls["AeEnable"] = True
                else:
                    controls["AeEnable"] = False

            controls["AwbEnable"] =  bool(autowb)

            if controls:
                # Best effort: set before and after start to ensure they stick
                self._set_controls(controls)

            self.picam2.start()
            self.cam_open = True

            if controls:
                ok = self._set_controls(controls)
                if ok and not self.log.full():
                    self.log.put_nowait((logging.INFO, f"PiCam2:Controls set {controls}"))

            if not self.log.full():
                self.log.put_nowait((logging.INFO, "PiCam2:Camera opened"))

        except Exception as exc:
            self.picam2 = None
            self.cam_open = False
            if not self.log.full():
                self.log.put_nowait((logging.CRITICAL, f"PiCam2:Failed to open camera: {exc}"))

    def close_cam(self):
        try:
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                except Exception:
                    pass
                try:
                    self.picam2.close()
                except Exception:
                    pass
        finally:
            self.picam2 = None
            self.cam_open = False
            self._camera_controls = None
            self._last_metadata = None

    def _capture_metadata_nolock(self) -> dict:
        """Return latest metadata from Picamera2.

        Caller must hold self.cam_lock.
        """
        if self.picam2 is None:
            return {}
        try:
            capture_metadata = getattr(self.picam2, "capture_metadata", None)
            if callable(capture_metadata):
                metadata = capture_metadata()
                return dict(metadata) if metadata is not None else {}
        except Exception:
            return {}
        return {}

    def _get_metadata(self) -> dict:
        if self._last_metadata is not None:
            return self._last_metadata
        if self.picam2 is None:
            return {}
        with self.cam_lock:
            self._last_metadata = self._capture_metadata_nolock()
            return self._last_metadata

    def _get_control(self, name: str):
        metadata = self._get_metadata()
        if name in metadata:
            return metadata.get(name)
        # Some controls may not show up in metadata; try reading again.
        if self.picam2 is None:
            return None
        with self.cam_lock:
            self._last_metadata = self._capture_metadata_nolock()
            return (self._last_metadata or {}).get(name)

    def _set_controls(self, controls: dict) -> bool:
        if self.picam2 is None:
            return False
        if not controls:
            return True
        try:
            with self.cam_lock:
                self.picam2.set_controls(controls)
            return True
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, f"PiCam2:Failed to set controls {controls}: {exc}"))
            return False

    def _map_fourcc(self, fourcc: str) -> str | None:
        """Map common FOURCC strings to Picamera2/libcamera format names."""
        if not fourcc:
            return None
        table = {
            "BGR3": "BGR888",
            "RGB3": "RGB888",
            "MJPG": "MJPEG",
            "YUY2": "YUYV",
            "YUYV": "YUYV",
            "YUV420": "YUV420",
            "YU12": "YUV420",  # planar 4:2:0
            "I420": "YUV420",  # planar 4:2:0
        }
        return table.get(fourcc)

    def _flip_to_transform(self, flip: int, Transform):
        """Map cv2 flip enum to libcamera Transform."""
        try:
            t = Transform()
        except Exception:
            return None
        if flip == 0:
            return t
        if flip == 1:   # ccw 90
            t = Transform(hflip=0, vflip=0, rotation=270)
        elif flip == 2: # 180
            t = Transform(hflip=1, vflip=1)
        elif flip == 3: # cw 90
            t = Transform(hflip=0, vflip=0, rotation=90)
        elif flip == 4: # horizontal
            t = Transform(hflip=1, vflip=0)
        elif flip == 6: # vertical
            t = Transform(hflip=0, vflip=1)
        elif flip == 5: # upright diagonal (ccw90 + hflip)
            t = Transform(hflip=1, vflip=0, rotation=270)
        elif flip == 7: # upper-left diagonal (transpose)
            t = Transform(hflip=1, vflip=0, rotation=90)
        return t
        
    # ---------------------------------------------------------------------
    # Picamera2/libcamera Getters & Setters
    # ---------------------------------------------------------------------

    @property
    def width(self):
        return int(self._capture_width)

    @width.setter
    def width(self, value):
        if value is None or value == -1:
            return
        self.size = (int(value), int(self._capture_height))

    @property
    def height(self):
        return int(self._capture_height)

    @height.setter
    def height(self, value):
        if value is None or value == -1:
            return
        self.size = (int(self._capture_width), int(value))

    @property
    def resolution(self):
        return (int(self._capture_width), int(self._capture_height))

    @resolution.setter
    def resolution(self, value):
        if value is None or value == -1:
            return
        if isinstance(value, (tuple, list)):
            if len(value) < 2:
                self.size = (int(value[0]), int(value[0]))
            else:
                self.size = (int(value[0]), int(value[1]))
        else:
            self.size = (int(value), int(value))

    @property
    def exposure(self):
        return self._get_control("ExposureTime")
    @exposure.setter
    def exposure(self, value):
        if value is None or value == -1:
            return
        self._exposure = float(value)
        self._set_controls({"AeEnable": False, "ExposureTime": int(value)})

    @property
    def gain(self):
        return self._get_control("AnalogueGain")
    @gain.setter
    def gain(self, value):
        if value is None:
            return
        self._set_controls({"AnalogueGain": float(value)})

    @property
    def autoexposure(self):
        # Match cv2Capture semantics: -1 unknown, else 0/1.
        ae = self._get_control("AeEnable")
        if ae is None:
            return -1
        return 1 if bool(ae) else 0
    @autoexposure.setter
    def autoexposure(self, value):
        if value is None or value == -1:
            return
        self._autoexposure = 1 if bool(value) else 0
        self._set_controls({"AeEnable": bool(value)})

    @property
    def autowb(self):
        awb = self._get_control("AwbEnable")
        if awb is None:
            return -1
        return 1 if bool(awb) else 0
    @autowb.setter
    def autowb(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AwbEnable": bool(value)})

    @property
    def wbtemperature(self):
        # Best-effort mapping: libcamera uses ColourTemperature.
        return self._get_control("ColourTemperature")

    @wbtemperature.setter
    def wbtemperature(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"ColourTemperature": int(value)})

    @property
    def colourgains(self):
        return self._get_control("ColourGains")

    @colourgains.setter
    def colourgains(self, value):
        if value is None:
            return
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("ColourGains must be a (red_gain, blue_gain) tuple")
        self._set_controls({"ColourGains": (float(value[0]), float(value[1]))})

    @property
    def afmode(self):
        return self._get_control("AfMode")
    @afmode.setter
    def afmode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AfMode": int(value)})

    @property
    def lensposition(self):
        return self._get_control("LensPosition")
    @lensposition.setter
    def lensposition(self, value):
        if value is None:
            return
        self._set_controls({"LensPosition": float(value)})

    @property
    def aemeteringmode(self):
        return self._get_control("AeMeteringMode")
    @aemeteringmode.setter
    def aemeteringmode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AeMeteringMode": int(value)})

    @property
    def aeexposuremode(self):
        return self._get_control("AeExposureMode")
    @aeexposuremode.setter
    def aeexposuremode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"AeExposureMode": int(value)})

    @property
    def brightness(self):
        return self._get_control("Brightness")
    @brightness.setter
    def brightness(self, value):
        if value is None:
            return
        self._set_controls({"Brightness": float(value)})

    @property
    def contrast(self):
        return self._get_control("Contrast")
    @contrast.setter
    def contrast(self, value):
        if value is None:
            return
        self._set_controls({"Contrast": float(value)})

    @property
    def saturation(self):
        return self._get_control("Saturation")
    @saturation.setter
    def saturation(self, value):
        if value is None:
            return
        self._set_controls({"Saturation": float(value)})

    @property
    def sharpness(self):
        return self._get_control("Sharpness")
    @sharpness.setter
    def sharpness(self, value):
        if value is None:
            return
        self._set_controls({"Sharpness": float(value)})

    @property
    def noisereductionmode(self):
        return self._get_control("NoiseReductionMode")
    @noisereductionmode.setter
    def noisereductionmode(self, value):
        if value is None or value == -1:
            return
        self._set_controls({"NoiseReductionMode": int(value)})

    @property
    def fps(self):
        fps = self._get_control("FrameRate")
        if fps is None:
            return float(self._framerate)
        return fps
    @fps.setter
    def fps(self, value):
        if value is None or value == -1:
            return
        self._framerate = float(value)
        self._set_controls({"FrameRate": float(value)})

    @property
    def size(self):
        return (self._capture_width, self._capture_height)

    @size.setter
    def size(self, value):
        if value is None:
            return
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("size must be a (width, height) tuple")

        width, height = int(value[0]), int(value[1])
        if width <= 0 or height <= 0:
            raise ValueError("size must be positive")

        # Reconfiguration while the capture thread is running is risky.
        if not self.stopped:
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, "PiCam2:size change requires restart; stop() first"))
            return

        self._capture_width = width
        self._capture_height = height
        self._camera_res = (width, height)

        # If camera is open, reconfigure it.
        if self.cam_open:
            self.close_cam()
            self.open_cam()


###############################################################################
# Testing
###############################################################################
if __name__ == '__main__':

    configs = {
        'camera_res'      : (1280, 720),    # any amera: Camera width & height
        'exposure'        : 10000,          # any camera: -1,0 = auto, 1...max=frame interval in microseconds
        'autoexposure'    : 0,              # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
        'fps'             : 60,             # any camera: 1/10, 15, 30, 40, 90, 120 overlocked
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 0,              # 0=norotation 
                                            # 1=ccw90deg 
                                            # 2=rotation180 
                                            # 3=cw90 
                                            # 4=horizontal 
                                            # 5=upright diagonal flip 
                                            # 6=vertical 
                                            # 7=uperleft diagonal flip
        'displayfps'       : 30             # frame rate for display server
    }

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")

    logger.log(logging.DEBUG, "Starting PiCamera2 capture")

    camera = piCamera2Capture(configs, camera_num=0)
    camera.start()

    cv2.namedWindow("Pi Camera 2", cv2.WINDOW_AUTOSIZE)

    try:
        while cv2.getWindowProperty("Pi Camera 2", 0) >= 0:
            try:
                (frame_time, frame) = camera.capture.get(timeout=0.25)
                cv2.imshow("Pi Camera 2", frame)
            except Exception:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            try:
                (level, msg) = camera.log.get_nowait()
                logger.log(level, f"PiCam2:{msg}")
            except Exception:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        try:
            camera.join(timeout=2.0)
        except Exception:
            pass
        try:
            camera.close_cam()
        except Exception:
            pass
        cv2.destroyAllWindows()