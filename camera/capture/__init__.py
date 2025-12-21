
# Keep imports optional so non-RaspberryPi environments can still import the package.

try:
	from .picapture import piCapture
except Exception:  # pragma: no cover
	piCapture = None  # type: ignore

try:
	from .picamera2capture import piCamera2Capture
except Exception:  # pragma: no cover
	piCamera2Capture = None  # type: ignore

try:
	from .libcamcapture import libcameraCapture
except Exception:  # pragma: no cover
	libcameraCapture = None  # type: ignore

