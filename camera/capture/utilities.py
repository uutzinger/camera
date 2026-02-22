###############################################################################
# Camera enumeration utilities (best-effort, cross-platform)
#
# Provides helpers to list camera devices and supported modes using platform
# tools when available (v4l2-ctl / ffmpeg / gst-device-monitor-1.0) and
# vendor SDKs (Picamera2, PySpin).
###############################################################################

"""
Raspberry Pi CSI: Picamera2 through libcamera
Jetson CSI: gstreamer: nvarguscamerasource
Linux USB: oepncv / ffmpeg / gstreamer through v4l2
Linux Spinnacker through PySpin
Linux Basler through pypylon
Windows ffmpeg through dshow
macOS ffmpeg through avfoundation

1 Detect OS.
2 Run native scan for that OS:
    Linux: V4L2 + GStreamer monitor
    Windows: DirectShow
    macOS: AVFoundation
3 Add platform‑specific specialized scans:
    Raspberry Pi: Picamera2/libcamera
    Jetson: GStreamer nvarguscamerasrc
4 Add vendor SDK scans:
    Spinnaker (Teledyne/FLIR)
    Pylon (Basler)

2) Need to ask someone with a Mac and someone with Windows to run this to make ffmpeg device list work:
Linux
v4l2-ctrl --list-devices
v4l2-ctrl --device <devnode> --list-formats-ext
gst-device-monitor-1.0

macOS
ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg -f avfoundation -list_formats true -i <index>

WIndows
ffmpeg -list_devices true -f dshow -i dummy
ffmpeg -list_options true -f dshow -i video="YOUR DEVICE NAME"


"""

from __future__ import annotations

import os
import re
import sys
import glob
import shutil
import subprocess
import pprint
from typing import Any


def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return True, out
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        output = getattr(exc, "output", "")
        return False, output or str(exc)


def _tool_missing(name: str, hint: str) -> dict:
    # Per user request, do not emit error dicts; callers should handle empty.
    return {}


def _unescape_gst_value(val: str) -> str:
    if val is None:
        return ""
    # Handle common escaped sequences from gst-device-monitor-1.0 output.
    def _hex_replace(match: re.Match) -> str:
        try:
            return chr(int(match.group(1), 16))
        except Exception:
            return match.group(0)

    val = re.sub(r"\\x([0-9a-fA-F]{2})", _hex_replace, val)
    val = val.replace("\\,", ",").replace("\\ ", " ").replace("\\(", "(").replace("\\)", ")")
    val = val.replace("\\/", "/").replace("\\:", ":")
    val = val.replace('\\"', '"').replace("\\\\", "\\")
    return val.strip().strip('"')


def _append_format(formats: dict, fourcc: str, width: int, height: int, fps: float | None) -> None:
    fmt = formats.setdefault(fourcc, {})
    size_key = (width, height)
    size = fmt.setdefault(size_key, [])
    if fps is not None and fps not in size:
        size.append(fps)


def _formats_map_to_list(formats: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fourcc, sizes in formats.items():
        size_list = []
        for (w, h), fps_list in sorted(sizes.items()):
            size_list.append({"width": w, "height": h, "fps": sorted(fps_list)})
        out.append({"fourcc": fourcc, "sizes": size_list})
    return out


# ---------------------------------------------------------------------------
# V4L2 (Linux)
# ---------------------------------------------------------------------------

def list_v4l2_devices() -> list[dict[str, Any]]:
    if not shutil.which("v4l2-ctl"):
        return []
    ok, out = _run_cmd(["v4l2-ctl", "--list-devices"])
    if not ok:
        return []
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in out.splitlines():
        if not line.strip():
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            current = {"name": line.strip(), "devnodes": []}
            devices.append(current)
        elif current is not None:
            node = line.strip()
            if node.startswith("/dev/video"):
                current["devnodes"].append(node)
    return devices


def _parse_v4l2_formats(out: str) -> list[dict[str, Any]]:
    formats: list[dict[str, Any]] = []
    cur_fmt: dict[str, Any] | None = None
    cur_size: dict[str, Any] | None = None
    fmt_re = re.compile(r"\[\d+\]:\s+'(?P<fourcc>\w+)'\s+\((?P<desc>.+)\)")
    size_re = re.compile(r"Size:\s+Discrete\s+(?P<w>\d+)x(?P<h>\d+)")
    fps_re = re.compile(r"Interval:\s+Discrete\s+[0-9.]+s\s+\((?P<fps>[0-9.]+)\s+fps\)")

    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = fmt_re.search(line)
        if m:
            cur_fmt = {
                "fourcc": m.group("fourcc"),
                "description": m.group("desc"),
                "sizes": [],
            }
            formats.append(cur_fmt)
            cur_size = None
            continue
        m = size_re.search(line)
        if m and cur_fmt is not None:
            cur_size = {
                "width": int(m.group("w")),
                "height": int(m.group("h")),
                "fps": [],
            }
            cur_fmt["sizes"].append(cur_size)
            continue
        m = fps_re.search(line)
        if m and cur_size is not None:
            try:
                cur_size["fps"].append(float(m.group("fps")))
            except ValueError:
                pass
    return formats


def list_v4l2_formats(devnode: str) -> dict[str, Any]:
    if not shutil.which("v4l2-ctl"):
        return {}
    ok, out = _run_cmd(["v4l2-ctl", "--device", devnode, "--list-formats-ext"])
    if not ok:
        return {}
    return {"formats": _parse_v4l2_formats(out)}


def list_v4l2_cameras(max_devices: int | None = None) -> list[dict[str, Any]]:
    devices = list_v4l2_devices()
    devnodes: list[str] = []
    devnode_to_name: dict[str, str] = {}
    if devices:
        for dev in devices:
            name = dev.get("name")
            for node in dev.get("devnodes", []):
                devnodes.append(node)
                if name and node not in devnode_to_name:
                    devnode_to_name[node] = name
    else:
        devnodes = sorted(glob.glob("/dev/video*"))

    if max_devices is not None:
        devnodes = devnodes[:max_devices]

    cameras: list[dict[str, Any]] = []
    for node in devnodes:
        try:
            idx = int(os.path.basename(node).replace("video", ""))
        except ValueError:
            idx = None
        entry = {
            "index": idx,
            "devnode": node,
            "backend": "v4l2",
            "name": devnode_to_name.get(node),
        }
        fmt = list_v4l2_formats(node)
        if fmt.get("formats"):
            entry["formats"] = fmt.get("formats", [])
        if entry.get("formats"):
            cameras.append(entry)
    return cameras


def v4l2_camera_names(cameras: list[dict[str, Any]]) -> list[str]:
    """Return unique camera names in encounter order."""
    seen: set[str] = set()
    names: list[str] = []
    for cam in cameras:
        name = cam.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def v4l2_devnodes_for_camera(cameras: list[dict[str, Any]], name: str) -> list[str]:
    return [c.get("devnode") for c in cameras if c.get("name") == name and c.get("devnode")]


def v4l2_fourcc_for_devnode(cameras: list[dict[str, Any]], devnode: str) -> list[str]:
    for cam in cameras:
        if cam.get("devnode") != devnode:
            continue
        formats = cam.get("formats") or []
        return [f.get("fourcc") for f in formats if f.get("fourcc")]
    return []


def v4l2_resolutions_for_devnode_fourcc(
    cameras: list[dict[str, Any]],
    devnode: str,
    fourcc: str,
) -> list[str]:
    """Return list like '1280x720,30' where fps is the max available for that size."""
    for cam in cameras:
        if cam.get("devnode") != devnode:
            continue
        for fmt in cam.get("formats") or []:
            if fmt.get("fourcc") != fourcc:
                continue
            out: list[str] = []
            for size in fmt.get("sizes") or []:
                w = size.get("width")
                h = size.get("height")
                fps_list = size.get("fps") or []
                fps = max(fps_list) if fps_list else None
                if w is None or h is None:
                    continue
                if fps is None:
                    out.append(f"{w}x{h}")
                else:
                    out.append(f"{w}x{h},{fps:g}")
            return out
    return []


# ---------------------------------------------------------------------------
# FFmpeg (Windows / macOS)
# ---------------------------------------------------------------------------

def list_ffmpeg_dshow_devices() -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {}
    ok, out = _run_cmd(["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
    if not ok:
        # ffmpeg returns non-zero, but output still contains device list
        pass
    if "Unknown input format: 'dshow'" in out:
        return {}
    devices: list[dict[str, Any]] = []
    in_video = False
    for line in out.splitlines():
        if "DirectShow video devices" in line:
            in_video = True
            continue
        if "DirectShow audio devices" in line:
            in_video = False
            continue
        if in_video:
            m = re.search(r"\"(.+?)\"", line)
            if m:
                devices.append({"index": len(devices), "name": m.group(1)})
    return {"devices": devices}


def list_ffmpeg_dshow_formats(device_name: str) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {}
    ok, out = _run_cmd(["ffmpeg", "-list_options", "true", "-f", "dshow", "-i", f"video={device_name}"])
    if not ok:
        # still return raw output for parsing
        pass
    if "Unknown input format: 'dshow'" in out:
        return {}
    formats: dict[str, dict[tuple[int, int], list[float]]] = {}
    opt_re = re.compile(
        r"(?:(?:pixel_format|vcodec)=(?P<fmt>\\w+))\\s+min s=(?P<w1>\\d+)x(?P<h1>\\d+)\\s+fps=(?P<fps1>[0-9.]+)\\s+max s=(?P<w2>\\d+)x(?P<h2>\\d+)\\s+fps=(?P<fps2>[0-9.]+)"
    )
    for line in out.splitlines():
        m = opt_re.search(line)
        if not m:
            continue
        fmt = m.group("fmt").upper()
        w1, h1 = int(m.group("w1")), int(m.group("h1"))
        w2, h2 = int(m.group("w2")), int(m.group("h2"))
        fps1 = float(m.group("fps1"))
        fps2 = float(m.group("fps2"))
        _append_format(formats, fmt, w1, h1, fps1)
        _append_format(formats, fmt, w2, h2, fps2)
    return {"formats": _formats_map_to_list(formats)}


def list_ffmpeg_dshow_cameras() -> dict[str, Any]:
    devices = list_ffmpeg_dshow_devices()
    if not devices:
        return devices
    cams: list[dict[str, Any]] = []
    for dev in devices.get("devices", []):
        name = dev.get("name")
        if not name:
            continue
        fmt = list_ffmpeg_dshow_formats(name)
        entry = {"name": name, "index": dev.get("index"), "backend": "ffmpeg-dshow"}
        if fmt.get("formats"):
            entry["formats"] = fmt.get("formats", [])
        if entry.get("formats"):
            cams.append(entry)
    return {"cameras": cams}


def list_ffmpeg_avfoundation_devices() -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {}
    ok, out = _run_cmd(["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""])
    if not ok:
        pass
    if "Unknown input format: 'avfoundation'" in out:
        return {}
    devices: list[dict[str, Any]] = []
    for line in out.splitlines():
        m = re.search(r"\[(\d+)\]\s+(.+)", line)
        if m:
            devices.append({"index": int(m.group(1)), "name": m.group(2)})
    return {"devices": devices}


def list_ffmpeg_avfoundation_formats(index: int) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {}
    ok, out = _run_cmd(["ffmpeg", "-f", "avfoundation", "-list_formats", "true", "-i", str(index)])
    if not ok:
        pass
    if "Unknown input format: 'avfoundation'" in out:
        return {}
    formats: dict[str, dict[tuple[int, int], list[float]]] = {}
    line_re = re.compile(r"(?P<w>\\d+)x(?P<h>\\d+)(?:@(?P<fps>[0-9.]+))?")
    for line in out.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        w, h = int(m.group("w")), int(m.group("h"))
        fps = float(m.group("fps")) if m.group("fps") else None
        _append_format(formats, "AVF", w, h, fps)
    return {"formats": _formats_map_to_list(formats)}


def list_ffmpeg_avfoundation_cameras() -> dict[str, Any]:
    devices = list_ffmpeg_avfoundation_devices()
    if not devices:
        return devices
    cams: list[dict[str, Any]] = []
    for dev in devices.get("devices", []):
        idx = dev.get("index")
        if idx is None:
            continue
        fmt = list_ffmpeg_avfoundation_formats(int(idx))
        entry = {"name": dev.get("name"), "index": idx, "backend": "ffmpeg-avfoundation"}
        if fmt.get("formats"):
            entry["formats"] = fmt.get("formats", [])
        if entry.get("formats"):
            cams.append(entry)
    return {"cameras": cams}


def list_ffmpeg_v4l2_formats(devnode: str) -> dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {}
    debug = os.environ.get("CAMERA_DEBUG_FFMPEG") == "1"
    ok, out = _run_cmd(["ffmpeg", "-f", "v4l2", "-list_formats", "all", "-i", devnode])
    if debug:
        print(f"[ffmpeg-v4l2] devnode={devnode} ok={ok}", file=sys.stderr)
        print(out, file=sys.stderr)
    if not ok:
        pass
    if "Unknown input format: 'v4l2'" in out:
        return {}
    formats: dict[str, dict[tuple[int, int], list[float]]] = {}
    line_re = re.compile(r"(?:Compressed|Raw)\s*:\s*(?P<pix>\w+)")
    size_re = re.compile(r"(\d+)x(\d+)")
    for line in out.splitlines():
        if "Compressed" not in line and "Raw" not in line:
            continue
        m = line_re.search(line)
        if not m:
            continue
        sizes = size_re.findall(line)
        if not sizes:
            continue
        pix = m.group("pix").upper()
        map_fmt = {"YUYV422": "YUYV", "YUYV": "YUYV", "MJPEG": "MJPG", "GREY": "GREY", "GRAY": "GREY"}
        fourcc = map_fmt.get(pix, pix)
        for w, h in sizes:
            _append_format(formats, fourcc, int(w), int(h), None)
    return {"formats": _formats_map_to_list(formats)}


def list_ffmpeg_v4l2_cameras(max_devices: int | None = None) -> dict[str, Any]:
    devnodes = sorted(glob.glob("/dev/video*"))
    if max_devices is not None:
        devnodes = devnodes[:max_devices]
    cameras: list[dict[str, Any]] = []
    for node in devnodes:
        try:
            idx = int(os.path.basename(node).replace("video", ""))
        except ValueError:
            idx = None
        fmt = list_ffmpeg_v4l2_formats(node)
        entry = {"index": idx, "devnode": node, "backend": "ffmpeg-v4l2"}
        if fmt.get("formats"):
            entry["formats"] = fmt.get("formats", [])
        if entry.get("formats"):
            cameras.append(entry)
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# GStreamer (optional)
# ---------------------------------------------------------------------------

def list_gstreamer_devices() -> dict[str, Any]:
    if not shutil.which("gst-device-monitor-1.0"):
        return {}
    ok, out = _run_cmd(["gst-device-monitor-1.0"])
    if not ok:
        return {}
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    mode = None
    for raw in out.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped == "Device found:":
            if current:
                devices.append(current)
            current = {"caps": [], "properties": {}}
            mode = None
            continue
        if current is None:
            continue
        if stripped.startswith("name"):
            current["name"] = _unescape_gst_value(stripped.split(":", 1)[1])
            continue
        if stripped.startswith("class"):
            current["class"] = _unescape_gst_value(stripped.split(":", 1)[1])
            current["media_class"] = current["class"].split("/", 1)[0].lower()
            continue
        if stripped.startswith("caps"):
            mode = "caps"
            parts = stripped.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                current["caps"].append(parts[1].strip())
            continue
        if stripped.startswith("properties:"):
            mode = "props"
            continue
        if stripped.startswith("gst-launch-1.0"):
            mode = None
            continue
        if mode == "caps" and stripped:
            current["caps"].append(stripped)
            continue
        if mode == "props" and stripped:
            if "=" in stripped:
                key, val = stripped.split("=", 1)
                current["properties"][key.strip()] = _unescape_gst_value(val.strip())

    if current:
        devices.append(current)

    # Parse caps into structured formats where possible.
    for dev in devices:
        caps_entries: list[dict[str, Any]] = []
        for cap in dev.get("caps", []):
            tokens = [t.strip() for t in cap.split(",")]
            if not tokens:
                continue
            entry: dict[str, Any] = {"media_type": tokens[0]}
            for token in tokens[1:]:
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                entry[k.strip()] = v.strip()
            caps_entries.append(entry)

        # Convert caps to formats (video only)
        formats: dict[str, dict[tuple[int, int], list[float]]] = {}
        format_descs: dict[str, str] = {}
        for entry in caps_entries:
            media = entry.get("media_type")
            if not media or not media.startswith("video") and not media.startswith("image"):
                continue
            fmt = entry.get("format")
            if media == "image/jpeg":
                fourcc = "MJPG"
                desc = "image/jpeg"
            else:
                fmt_upper = fmt.upper() if isinstance(fmt, str) else ""
                map_fmt = {"YUY2": "YUYV", "GRAY8": "GREY", "GREY": "GREY"}
                fourcc = map_fmt.get(fmt_upper, fmt_upper or "RAW")
                desc = fmt_upper or "RAW"
            try:
                w = int(entry.get("width"))
                h = int(entry.get("height"))
            except Exception:
                continue
            fps_val = entry.get("framerate") or entry.get("fps")
            fps = None
            if isinstance(fps_val, str) and "/" in fps_val:
                try:
                    num, den = fps_val.split("/", 1)
                    fps = float(num) / float(den)
                except Exception:
                    fps = None
            elif fps_val:
                try:
                    fps = float(fps_val)
                except Exception:
                    fps = None
            _append_format(formats, fourcc, w, h, fps)
            if desc:
                format_descs.setdefault(fourcc, desc)

        if formats:
            fmt_list: list[dict[str, Any]] = []
            for fourcc, sizes in formats.items():
                size_list = []
                for (w, h), fps_list in sorted(sizes.items()):
                    size_list.append({"width": w, "height": h, "fps": sorted(fps_list)})
                item = {"fourcc": fourcc, "sizes": size_list}
                if fourcc in format_descs:
                    item["description"] = format_descs[fourcc]
                fmt_list.append(item)
            dev["formats"] = fmt_list

        # Promote devnode/index if present in properties
        devnode = dev.get("properties", {}).get("device.path")
        if devnode and devnode.startswith("/dev/video"):
            dev["devnode"] = devnode
            try:
                dev["index"] = int(os.path.basename(devnode).replace("video", ""))
            except Exception:
                pass

    # Drop non-video devices (no formats)
    devices = [d for d in devices if d.get("formats")]
    # Condense output: keep name/devnode/index/formats only.
    condensed: list[dict[str, Any]] = []
    for dev in devices:
        entry = {
            "name": dev.get("name"),
            "devnode": dev.get("devnode"),
            "index": dev.get("index"),
            "node": dev.get("properties", {}).get("node.name"),
            "formats": dev.get("formats", []),
        }
        if entry.get("formats"):
            condensed.append(entry)
    return {"devices": condensed}


def _gstreamer_device_list(devices: Any) -> list[dict[str, Any]]:
    if isinstance(devices, dict):
        return list(devices.get("devices") or [])
    if isinstance(devices, list):
        return list(devices)
    return []


def gstreamer_camera_names(devices: Any) -> list[str]:
    """Return unique camera names in encounter order."""
    out: list[str] = []
    seen: set[str] = set()
    for dev in _gstreamer_device_list(devices):
        name = dev.get("name") or dev.get("devnode") or dev.get("node")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def gstreamer_devnodes_for_camera(devices: Any, name: str) -> list[str]:
    devnodes: list[str] = []
    for dev in _gstreamer_device_list(devices):
        dev_name = dev.get("name") or dev.get("devnode") or dev.get("node")
        if dev_name != name:
            continue
        devnode = dev.get("devnode")
        if devnode:
            devnodes.append(devnode)
    return devnodes


def gstreamer_fourcc_for_devnode(devices: Any, devnode: str) -> list[str]:
    for dev in _gstreamer_device_list(devices):
        if dev.get("devnode") != devnode:
            continue
        formats = dev.get("formats") or []
        return [f.get("fourcc") for f in formats if f.get("fourcc")]
    return []


def gstreamer_resolutions_for_devnode_fourcc(
    devices: Any,
    devnode: str,
    fourcc: str,
) -> list[str]:
    """Return list like '1280x720,30' where fps is the max available for that size."""
    for dev in _gstreamer_device_list(devices):
        if dev.get("devnode") != devnode:
            continue
        for fmt in dev.get("formats") or []:
            if fmt.get("fourcc") != fourcc:
                continue
            out: list[str] = []
            for size in fmt.get("sizes") or []:
                w = size.get("width")
                h = size.get("height")
                fps_list = size.get("fps") or []
                fps = max(fps_list) if fps_list else None
                if w is None or h is None:
                    continue
                if fps is None:
                    out.append(f"{w}x{h}")
                else:
                    out.append(f"{w}x{h},{fps:g}")
            return out
    return []


# ---------------------------------------------------------------------------
# Picamera2 (Raspberry Pi)
# ---------------------------------------------------------------------------

def list_picamera2_modes() -> dict[str, Any]:
    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception as exc:
        return {}

    cam = Picamera2()
    props = getattr(cam, "camera_properties", {}) or {}
    modes = getattr(cam, "sensor_modes", []) or []
    out_modes: list[dict[str, Any]] = []
    for m in modes:
        if not isinstance(m, dict):
            out_modes.append({"raw": m})
            continue
        size = m.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            w, h = int(size[0]), int(size[1])
        else:
            w = h = None
        out_modes.append(
            {
                "width": w,
                "height": h,
                "format": m.get("format"),
                "bit_depth": m.get("bit_depth"),
                "fps": m.get("fps"),
            }
        )
    cam.close()
    name = None
    if isinstance(props, dict):
        name = props.get("Model") or props.get("ModelName") or props.get("PixelArraySize")
    return {"index": 0, "name": name, "backend": "picamera2", "properties": props, "modes": out_modes}


# ---------------------------------------------------------------------------
# Spinnaker (PySpin)
# ---------------------------------------------------------------------------

def list_spin_cameras() -> dict[str, Any]:
    try:
        import PySpin  # type: ignore
    except Exception as exc:
        return {}

    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num = cam_list.GetSize()
    cameras: list[dict[str, Any]] = []
    for idx in range(num):
        cam = cam_list.GetByIndex(idx)
        entry: dict[str, Any] = {"index": idx, "backend": "pyspin"}
        try:
            cam.Init()
            nodemap = cam.GetTLDeviceNodeMap()
            for name in ("DeviceModelName", "DeviceSerialNumber", "DeviceVendorName", "DeviceVersion"):
                node = PySpin.CStringPtr(nodemap.GetNode(name))
                if node and PySpin.IsReadable(node):
                    entry[name] = node.GetValue()
            if "DeviceModelName" in entry:
                entry["name"] = entry.get("DeviceModelName")
            # Enumerate PixelFormat entries if available
            try:
                pf_node = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode("PixelFormat"))
                if pf_node and PySpin.IsReadable(pf_node):
                    entries = pf_node.GetEntries()
                    formats = []
                    for e in entries:
                        if PySpin.IsReadable(e):
                            formats.append(e.GetSymbolic())
                    entry["pixel_formats"] = formats
            except Exception:
                pass
        except Exception as exc:
            entry["error"] = str(exc)
        finally:
            try:
                cam.DeInit()
            except Exception:
                pass
        if entry.get("pixel_formats"):
            cameras.append(entry)
    cam_list.Clear()
    system.ReleaseInstance()
    return {"cameras": cameras}


# ---------------------------------------------------------------------------
# Cross-platform convenience
# ---------------------------------------------------------------------------

def list_camera_modes() -> dict[str, Any]:
    """Return best-effort camera modes across available backends."""
    plat = sys.platform
    backends: dict[str, Any] = {}

    v4l2 = list_v4l2_cameras()
    if v4l2:
        backends["v4l2"] = v4l2

    ff_v4l2 = list_ffmpeg_v4l2_cameras()
    if ff_v4l2.get("cameras"):
        backends["ffmpeg-v4l2"] = ff_v4l2

    gst = list_gstreamer_devices()
    if gst.get("devices"):
        backends["gstreamer"] = gst

    dshow = list_ffmpeg_dshow_cameras()
    if dshow.get("cameras"):
        backends["ffmpeg-dshow"] = dshow

    avf = list_ffmpeg_avfoundation_cameras()
    if avf.get("cameras"):
        backends["ffmpeg-avfoundation"] = avf

    picam = list_picamera2_modes()
    if picam.get("modes"):
        backends["picamera2"] = picam

    spin = list_spin_cameras()
    if spin.get("cameras"):
        backends["pyspin"] = spin

    return {"platform": plat, "backends": backends}


def list_blackfly_cameras() -> dict[str, Any]:
    """Deprecated alias for list_spin_cameras()."""
    return list_spin_cameras()


__all__ = [
    "list_v4l2_devices",
    "list_v4l2_formats",
    "list_v4l2_cameras",
    "v4l2_camera_names",
    "v4l2_devnodes_for_camera",
    "v4l2_fourcc_for_devnode",
    "v4l2_resolutions_for_devnode_fourcc",
    "list_ffmpeg_dshow_devices",
    "list_ffmpeg_dshow_formats",
    "list_ffmpeg_dshow_cameras",
    "list_ffmpeg_avfoundation_devices",
    "list_ffmpeg_avfoundation_formats",
    "list_ffmpeg_avfoundation_cameras",
    "list_ffmpeg_v4l2_formats",
    "list_ffmpeg_v4l2_cameras",
    "list_gstreamer_devices",
    "gstreamer_camera_names",
    "gstreamer_devnodes_for_camera",
    "gstreamer_fourcc_for_devnode",
    "gstreamer_resolutions_for_devnode_fourcc",
    "list_picamera2_modes",
    "list_spin_cameras",
    "list_blackfly_cameras",
    "list_camera_modes",
]


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")

def _pp(obj: Any) -> None:
    pprint.pprint(obj, width=120, compact=False, sort_dicts=False)


def main() -> None:
    _print_section("V4L2 Cameras")
    cameras = list_v4l2_cameras()
    # _pp(cameras)
    _print_section("V4L2 First Camera")
    names = v4l2_camera_names(cameras)
    _pp(names)
    devnodes = v4l2_devnodes_for_camera(cameras, names[0] if names else "")
    _pp(devnodes)
    fourccs = v4l2_fourcc_for_devnode(cameras, devnodes[0] if devnodes else "")
    _pp(fourccs)
    resolutions = v4l2_resolutions_for_devnode_fourcc(cameras, devnodes[0] if devnodes else "", fourccs[0] if fourccs else "")
    _pp(resolutions)

    _print_section("V4L2 Second Camera")
    names = v4l2_camera_names(cameras)
    _pp(names)
    devnodes = v4l2_devnodes_for_camera(cameras, names[0] if names else "")
    _pp(devnodes)
    fourccs = v4l2_fourcc_for_devnode(cameras, devnodes[1] if devnodes else "")
    _pp(fourccs)
    resolutions = v4l2_resolutions_for_devnode_fourcc(cameras, devnodes[1] if devnodes else "", fourccs[0] if fourccs else "")
    _pp(resolutions)


    _print_section("GStreamer First Camera")
    gst_devices = list_gstreamer_devices()
    gst_names = gstreamer_camera_names(gst_devices)
    _pp(gst_names)
    gst_devnodes = gstreamer_devnodes_for_camera(gst_devices, gst_names[0] if gst_names else "")
    _pp(gst_devnodes)
    gst_fourccs = gstreamer_fourcc_for_devnode(gst_devices, gst_devnodes[0] if gst_devnodes else "")
    _pp(gst_fourccs)
    gst_resolutions = gstreamer_resolutions_for_devnode_fourcc(
        gst_devices,
        gst_devnodes[0] if gst_devnodes else "",
        gst_fourccs[0] if gst_fourccs else "",
    )
    _pp(gst_resolutions)

    _print_section("GStreamer Second Camera")
    gst_devices = list_gstreamer_devices()
    gst_names = gstreamer_camera_names(gst_devices)
    _pp(gst_names)
    gst_devnodes = gstreamer_devnodes_for_camera(gst_devices, gst_names[1] if len(gst_names) > 1 else "")
    _pp(gst_devnodes)
    gst_fourccs = gstreamer_fourcc_for_devnode(gst_devices, gst_devnodes[0] if gst_devnodes else "")
    _pp(gst_fourccs)
    gst_resolutions = gstreamer_resolutions_for_devnode_fourcc(
        gst_devices,
        gst_devnodes[0] if gst_devnodes else "",
        gst_fourccs[0] if gst_fourccs else "",
    )
    _pp(gst_resolutions)


    _print_section("FFmpeg DirectShow Devices (Windows)")
    _pp(list_ffmpeg_dshow_devices())

    _print_section("FFmpeg DirectShow Cameras (Windows)")
    _pp(list_ffmpeg_dshow_cameras())

    _print_section("FFmpeg AVFoundation Devices (macOS)")
    _pp(list_ffmpeg_avfoundation_devices())

    _print_section("FFmpeg AVFoundation Cameras (macOS)")
    _pp(list_ffmpeg_avfoundation_cameras())

    _print_section("FFmpeg v4l2 Cameras (Linux)")
    _pp(list_ffmpeg_v4l2_cameras())

    _print_section("GStreamer Devices")
    _pp(list_gstreamer_devices())

    _print_section("Picamera2 Modes")
    _pp(list_picamera2_modes())

    _print_section("Spinnaker Cameras")
    _pp(list_spin_cameras())

    _print_section("Cross-Platform Camera Modes")
    _pp(list_camera_modes())


if __name__ == "__main__":
    main()
