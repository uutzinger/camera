"""Legacy libcamera capture backend.

This module is kept for backward compatibility.

The original implementation lives in `camera.capture.archive.libcamcapture`.
Prefer using `camera.capture.picamera2capture.piCamera2Capture` on Raspberry Pi
or `camera.capture.gcapture.gCapture` where appropriate.
"""

from __future__ import annotations

import warnings


warnings.warn(
    "camera.capture.libcamcapture is deprecated; use picamera2capture or gcapture instead",
    DeprecationWarning,
    stacklevel=2,
)


from .archive.libcamcapture import libcameraCapture  # noqa: E402


__all__ = ["libcameraCapture"]
