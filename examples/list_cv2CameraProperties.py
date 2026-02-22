##########################################################################
# List opencv camera properties (concise by default)
##########################################################################
import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys

# Suppress OpenCV backend warnings unless explicitly enabled.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2

# Setting up logging
logging.basicConfig(level=logging.INFO)  # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("CV2")

def decode_fourcc(val):
    """ decode the fourcc integer to the chracter string """
    return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

def cv2_settings_summary(capture, verbose: bool) -> None:
    """Print concise or verbose OpenCV camera properties."""
    if not capture.isOpened():
        logger.info("  (not opened)")
        return

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc_val = capture.get(cv2.CAP_PROP_FOURCC)
    fourcc = decode_fourcc(fourcc_val) if fourcc_val not in (0, -1) else ""
    backend = capture.get(cv2.CAP_PROP_BACKEND)

    msg = f"  {int(width)}x{int(height)} @ {fps:.1f} fps"
    if fourcc:
        msg += f", fourcc={fourcc}"
    if backend:
        msg += f", backend={int(backend)}"
    logger.info(msg)

    if not verbose:
        return

    props = [
        ("POS_MSEC", cv2.CAP_PROP_POS_MSEC),
        ("POS_FRAMES", cv2.CAP_PROP_POS_FRAMES),
        ("AVI_RATIO", cv2.CAP_PROP_POS_AVI_RATIO),
        ("FRAME_CNT", cv2.CAP_PROP_FRAME_COUNT),
        ("FORMAT", cv2.CAP_PROP_FORMAT),
        ("MODE", cv2.CAP_PROP_MODE),
        ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
        ("CONTRAST", cv2.CAP_PROP_CONTRAST),
        ("SATURATION", cv2.CAP_PROP_SATURATION),
        ("HUE", cv2.CAP_PROP_HUE),
        ("GAIN", cv2.CAP_PROP_GAIN),
        ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
        ("CONV_RGB", cv2.CAP_PROP_CONVERT_RGB),
        ("AUTO_EXP", cv2.CAP_PROP_AUTO_EXPOSURE),
        ("GAMMA", cv2.CAP_PROP_GAMMA),
        ("AUTOWB", cv2.CAP_PROP_AUTO_WB),
        ("WB_TEMP", cv2.CAP_PROP_WB_TEMPERATURE),
        ("FOCUS", cv2.CAP_PROP_FOCUS),
        ("AUTOFOCUS", cv2.CAP_PROP_AUTOFOCUS),
        ("BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE),
        ("SAR_NUM", cv2.CAP_PROP_SAR_NUM),
        ("SAR_DEN", cv2.CAP_PROP_SAR_DEN),
        ("CHANNEL", cv2.CAP_PROP_CHANNEL),
    ]

    for name, prop in props:
        try:
            val = capture.get(prop)
        except Exception:
            continue
        logger.info(f"    {name}: {val}")

def _print_tool_missing(tool: str, hint: str) -> None:
    logger.info("  (%s not installed; %s)", tool, hint)


def v4l2_list_formats(devnode: str) -> None:
    """Print formats/resolutions from v4l2-ctl if available."""
    if not shutil.which("v4l2-ctl"):
        _print_tool_missing("v4l2-ctl", "install v4l-utils")
        return
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--device", devnode, "--list-formats-ext"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.info("  (v4l2-ctl failed for %s)", devnode)
        if exc.output:
            for line in exc.output.strip().splitlines():
                logger.info("  %s", line)
        return
    for line in out.strip().splitlines():
        logger.info("  %s", line)


def ffmpeg_list_formats_windows(index: int) -> None:
    """List formats on Windows via ffmpeg + DirectShow."""
    if not shutil.which("ffmpeg"):
        _print_tool_missing("ffmpeg", "install ffmpeg and ensure it is on PATH")
        return
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        out = exc.output or ""
    devices = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            devices.append(line.strip('"'))
        elif " \"".encode() in b"":
            pass
        elif " \"" in line:
            devices.append(line.split(" \"", 1)[1].rstrip('"'))
    if index >= len(devices):
        logger.info("  (no DirectShow device for index %s)", index)
        return
    name = devices[index]
    logger.info("  DirectShow device: %s", name)
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-list_options", "true", "-f", "dshow", "-i", f"video={name}"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        out = exc.output or ""
    for line in out.strip().splitlines():
        logger.info("  %s", line)


def ffmpeg_list_formats_macos(index: int) -> None:
    """List formats on macOS via ffmpeg + AVFoundation."""
    if not shutil.which("ffmpeg"):
        _print_tool_missing("ffmpeg", "install ffmpeg and ensure it is on PATH")
        return
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-f", "avfoundation", "-list_formats", "true", "-i", str(index)],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        out = exc.output or ""
    for line in out.strip().splitlines():
        logger.info("  %s", line)


def list_formats_cross_platform(index: int, devnode: str) -> None:
    plat = sys.platform
    if plat.startswith("linux"):
        v4l2_list_formats(devnode)
    elif plat == "win32":
        ffmpeg_list_formats_windows(index)
    elif plat == "darwin":
        ffmpeg_list_formats_macos(index)
    else:
        logger.info("  (format listing not supported on platform: %s)", plat)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=5, help="Number of indices to probe")
    parser.add_argument("--verbose", action="store_true", help="Show detailed properties")
    parser.add_argument("--very-verbose", action="store_true", help="Show all available properties")
    parser.add_argument("--opencv-warnings", action="store_true", help="Allow OpenCV backend warnings")
    parser.add_argument("--list-formats", action="store_true", help="List all V4L2 formats/resolutions")
    parser.add_argument(
        "--no-list-formats",
        action="store_true",
        help="Disable V4L2 format listing (default is on)",
    )
    args = parser.parse_args()

    if args.opencv_warnings:
        if "OPENCV_LOG_LEVEL" in os.environ:
            os.environ.pop("OPENCV_LOG_LEVEL")
    else:
        for level in ("LOG_LEVEL_SILENT", "LOG_LEVEL_ERROR", "LOG_LEVEL_WARNING"):
            if hasattr(cv2, level):
                cv2.setLogLevel(getattr(cv2, level))
                break

    # Probe existing /dev/video* nodes to avoid backend errors for non-devices.
    dev_nodes = sorted(glob.glob("/dev/video*"))
    indices = []
    for node in dev_nodes:
        try:
            idx = int(os.path.basename(node).replace("video", ""))
        except ValueError:
            continue
        indices.append(idx)
    if not indices:
        indices = list(range(args.max))
    else:
        indices = indices[: args.max]

    for index in indices:
        devnode = f"/dev/video{index}"
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            logger.info("Camera %s:", index)
            cv2_settings_summary(cap, args.verbose or args.very_verbose)
            if args.very_verbose:
                props_full = [
                    ("POS_MSEC", cv2.CAP_PROP_POS_MSEC),
                    ("POS_FRAMES", cv2.CAP_PROP_POS_FRAMES),
                    ("AVI_RATIO", cv2.CAP_PROP_POS_AVI_RATIO),
                    ("WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
                    ("HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
                    ("FPS", cv2.CAP_PROP_FPS),
                    ("FOURCC", cv2.CAP_PROP_FOURCC),
                    ("FRAME_CNT", cv2.CAP_PROP_FRAME_COUNT),
                    ("FORMAT", cv2.CAP_PROP_FORMAT),
                    ("MODE", cv2.CAP_PROP_MODE),
                    ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
                    ("CONTRAST", cv2.CAP_PROP_CONTRAST),
                    ("SATURATION", cv2.CAP_PROP_SATURATION),
                    ("HUE", cv2.CAP_PROP_HUE),
                    ("GAIN", cv2.CAP_PROP_GAIN),
                    ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
                    ("CONV_RGB", cv2.CAP_PROP_CONVERT_RGB),
                    ("RECT", cv2.CAP_PROP_RECTIFICATION),
                    ("MONO", cv2.CAP_PROP_MONOCHROME),
                    ("SHARP", cv2.CAP_PROP_SHARPNESS),
                    ("AUTO_EXP", cv2.CAP_PROP_AUTO_EXPOSURE),
                    ("GAMMA", cv2.CAP_PROP_GAMMA),
                    ("TRIGGER", cv2.CAP_PROP_TRIGGER),
                    ("TRIGGER_DEL", cv2.CAP_PROP_TRIGGER_DELAY),
                    ("AUTOWB", cv2.CAP_PROP_AUTO_WB),
                    ("WB_TEMP", cv2.CAP_PROP_WB_TEMPERATURE),
                    ("WB_BLUE", cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
                    ("WB_RED", cv2.CAP_PROP_WHITE_BALANCE_RED_V),
                    ("TEMP", cv2.CAP_PROP_TEMPERATURE),
                    ("ZOOM", cv2.CAP_PROP_ZOOM),
                    ("FOCUS", cv2.CAP_PROP_FOCUS),
                    ("GUID", cv2.CAP_PROP_GUID),
                    ("ISO", cv2.CAP_PROP_ISO_SPEED),
                    ("BACKLIGHT", cv2.CAP_PROP_BACKLIGHT),
                    ("PAN", cv2.CAP_PROP_PAN),
                    ("TILT", cv2.CAP_PROP_TILT),
                    ("ROLL", cv2.CAP_PROP_ROLL),
                    ("IRIS", cv2.CAP_PROP_IRIS),
                    ("SETTINGS", cv2.CAP_PROP_SETTINGS),
                    ("BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE),
                    ("AUTOFOCUS", cv2.CAP_PROP_AUTOFOCUS),
                    ("SAR_NUM", cv2.CAP_PROP_SAR_NUM),
                    ("SAR_DEN", cv2.CAP_PROP_SAR_DEN),
                    ("BACKEND", cv2.CAP_PROP_BACKEND),
                    ("CHANNEL", cv2.CAP_PROP_CHANNEL),
                ]
                for name, prop in props_full:
                    try:
                        val = cap.get(prop)
                    except Exception:
                        continue
                    if name == "FOURCC":
                        fourcc = decode_fourcc(val)
                        logger.info(f"    {name}: {val}")
                        if fourcc:
                            logger.info(f"    {name}_STR: {fourcc}")
                    else:
                        logger.info(f"    {name}: {val}")
            if args.list_formats or not args.no_list_formats:
                list_formats_cross_platform(index, devnode)
        cap.release()


if __name__ == "__main__":
    main()
