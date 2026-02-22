###############################################################################
# Jetson Nano CSI capture wrapper (GStreamer)
#
# Jetson-focused wrapper around gCapture.
# - Defaults source selection to nvarguscamerasrc
# - Keeps legacy nanoCapture class name/API
# - Inherits GI + OpenCV CAP_GSTREAMER fallback from gCapture
#
# Urs Utzinger
# GPT-5.2
###############################################################################

from __future__ import annotations

import logging

from .gcapture import gCapture


class nanoCapture(gCapture):
    """
    Jetson-focused compatibility wrapper around gCapture.

    This keeps legacy import paths (`camera.capture.nanocapture.nanoCapture`) while
    reusing the shared GStreamer implementation in gCapture.
    """

    def __init__(
        self,
        configs,
        camera_num: int = 0,
        res: tuple | None = None,
        exposure: float | None = None,
        queue_size: int = 32,
    ):
        merged = dict(configs or {})

        # Preserve nanoCapture behavior: nvargus by default on Jetson.
        if not merged.get("gst_source") and not merged.get("source"):
            merged["gst_source"] = "nvargus"

        # Preserve prior default capture size used by nanoCapture.
        merged.setdefault("camera_res", (1280, 720))

        # Preserve legacy auto-exposure range hint when no explicit source props
        # are provided and the caller is using nvargus in auto-exposure mode.
        src_name = str(merged.get("gst_source", merged.get("source", "")) or "").strip().lower()
        has_props = bool(str(merged.get("gst_source_props_str", "") or "").strip())
        try:
            exp_cfg = float(merged.get("exposure", -1))
        except Exception:
            exp_cfg = -1.0
        exp_eff = float(exposure) if exposure is not None else exp_cfg
        if src_name in ("nvargus", "nvarguscamerasrc") and (not has_props) and exp_eff <= 0:
            merged["gst_source_props_str"] = 'aelock=false exposuretimerange="34000 358733000"'

        super().__init__(
            merged,
            camera_num=camera_num,
            res=res,
            exposure=exposure,
            queue_size=queue_size,
        )

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, val):
        if val is None or val == -1:
            return

        self._exposure = float(val)
        src = getattr(self, "source", None)
        if src is None:
            return

        try:
            exposure_ns = int(float(val) * 1000)
            with self.cam_lock:
                if src.find_property("exposuretimerange") is not None:
                    src.set_property("exposuretimerange", f"{exposure_ns} {exposure_ns}")
                if src.find_property("aelock") is not None:
                    src.set_property("aelock", True)
            if not self.log.full():
                self.log.put_nowait((logging.INFO, f"NanoCap:Exposure set:{val}"))
        except Exception as exc:
            if not self.log.full():
                self.log.put_nowait((logging.WARNING, f"NanoCap:Failed to set exposure:{val}: {exc}"))
