#!/usr/bin/env python3
"""Direct Picamera2 main-stream FPS probe.

Purpose:
- Configure Picamera2 "main" stream directly (no project wrapper)
- Grab frames in a tight loop
- Print measured FPS periodically

Typical use:
  python3 examples/picamera2_main_fps_direct.py
  python3 examples/picamera2_main_fps_direct.py --size 640x480
  python3 examples/picamera2_main_fps_direct.py --fps 30

Notes:
- Requesting FPS above what the sensor mode supports may be ignored or behave poorly.
- For forcing frame timing, libcamera often uses FrameDurationLimits (microseconds).
"""

from __future__ import annotations

import argparse
import os
import time


def _parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Size must be like 640x480")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Size must be positive")
    return (w, h)

def _safe_get(obj, attr_name: str):
    try:
        return getattr(obj, attr_name)
    except Exception:
        return None

def _print_camera_config_and_controls(picam2, args, controls) -> None:
    print("=== camera configuration ===")
    try:
        print(
            "Requested: mode=main size={} format={} fps={} stream_policy=default low_latency=False flip=0".format(
                args.size, args.format, args.fps
            )
        )
    except Exception:
        pass
    try:
        print(f"Requested controls: {controls}")
    except Exception:
        pass
    try:
        cfg_now = picam2.camera_configuration()
        print(f"camera_configuration: {cfg_now}")
    except Exception as exc:
        print(f"camera_configuration unavailable ({exc})")
    try:
        props = _safe_get(picam2, "camera_properties")
        if props is not None:
            print(f"camera_properties: {props}")
    except Exception:
        pass

    # Read back timing-related controls/metadata (best-effort)
    try:
        md = picam2.capture_metadata()
        if isinstance(md, dict):
            fd = md.get("FrameDuration")
            fdl = md.get("FrameDurationLimits")
            sc = md.get("ScalerCrop")
            ae = md.get("AeEnable", None)
            exp = md.get("ExposureTime", None)
            print(
                "metadata: FrameDuration={} FrameDurationLimits={} ScalerCrop={} AeEnable={} ExposureTime={}".format(
                    fd, fdl, sc, ae, exp
                )
            )
    except Exception as exc:
        print(f"capture_metadata unavailable ({exc})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Direct Picamera2 main-stream FPS probe (no wrapper)")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--size", type=_parse_size, default=(640, 480), help="Main stream size WxH")
    ap.add_argument(
        "--format",
        type=str,
        default="BGR888",
        help="Main stream format (e.g. BGR888, RGB888, XRGB8888, XBGR8888, YUV420)",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Requested FPS. 0 = do not request. Uses FrameDurationLimits when possible.",
    )
    ap.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display frames in an OpenCV window (default: True). Use --no-display to disable.",
    )
    ap.add_argument(
        "--display-fps",
        type=float,
        default=10.0,
        help="Max display refresh rate (Hz). 0 = update every captured frame.",
    )
    ap.add_argument("--duration", type=float, default=15.0, help="Total run time in seconds")
    ap.add_argument("--warmup", type=float, default=1.0, help="Warmup time in seconds (not counted)")
    ap.add_argument("--report", type=float, default=2.0, help="Report interval in seconds")
    ap.add_argument("--list-modes", action="store_true", help="Print sensor modes and exit")
    args = ap.parse_args()

    # Reduce libcamera logging noise (optional)
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")  # 3=ERROR

    try:
        from picamera2 import Picamera2
    except Exception as exc:
        print(f"ERROR: picamera2 import failed: {exc}")
        return 2

    cv2 = None
    if args.display:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            print(f"WARNING: OpenCV (cv2) not available; disabling display: {exc}")
            args.display = False

    picam2 = Picamera2(camera_num=args.camera)

    if args.list_modes:
        modes = getattr(picam2, "sensor_modes", None)
        print("=== sensor_modes ===")
        if isinstance(modes, list):
            for i, m in enumerate(modes):
                try:
                    size = m.get("size")
                    fmt = m.get("format")
                    fps = m.get("fps", m.get("max_fps", None))
                    print(f"[{i}] size={size} format={fmt} fps={fps}")
                except Exception:
                    print(f"[{i}] {m}")
        else:
            print(modes)
        return 0

    # Always pass a dict for `controls`.
    # Some Picamera2/libcamera builds have issues when controls=None (e.g. internal
    # dict-merge with None, or defaults that reference enums not available on older stacks).
    controls: dict[str, object] = {}

    # Request timing (best-effort). FrameDurationLimits is (min_us, max_us).
    if args.fps and args.fps > 0:
        frame_us = int(round(1_000_000.0 / float(args.fps)))
        controls["FrameDurationLimits"] = (frame_us, frame_us)

    # Build configuration
    cfg = picam2.create_video_configuration(
        main={"size": args.size, "format": args.format},
        controls=controls,
    )
    picam2.configure(cfg)

    # Start camera
    picam2.start()

    # Apply controls after start as well (some pipelines only accept runtime control changes)
    if controls:
        try:
            picam2.set_controls(controls)
            print(f"Controls set: {controls}")
        except Exception as exc:
            print(f"WARNING: set_controls({controls}) failed: {exc}")

    # Display setup (optional) - create the window early so failures are visible.
    window_name = "Picamera2 main (direct)"
    display_interval = 0.0
    if args.display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            # Pump the UI event loop once to ensure the window actually appears.
            cv2.waitKey(1)
        except Exception as exc:
            print(f"WARNING: OpenCV window init failed; disabling display: {exc}")
            args.display = False

        try:
            display_fps = float(args.display_fps or 0.0)
            display_interval = 0.0 if display_fps <= 0 else (1.0 / display_fps)
        except Exception:
            display_interval = 0.0

    # Warmup
    warmup_end = time.perf_counter() + float(args.warmup)
    while time.perf_counter() < warmup_end:
        _ = picam2.capture_array("main")

    _print_camera_config_and_controls(picam2, args, controls)

    # Measurement loop
    t0 = time.perf_counter()
    t_end = t0 + float(args.duration)
    t_report = t0 + float(args.report)

    # Display window already initialized above.
    last_display_time = 0.0
    window_check_enabled = True

    frames = 0
    last_report_frames = 0
    last_report_time = t0

    try:
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            frame = picam2.capture_array("main")
            frames += 1

            if args.display and frame is not None:
                if display_interval <= 0.0 or (now - last_display_time) >= display_interval:
                    frame_show = frame
                    # Convert to OpenCV BGR if needed.
                    fmt_u = str(args.format or "").upper()
                    try:
                        if frame_show.ndim == 3 and frame_show.shape[2] == 4:
                            frame_show = frame_show[:, :, :3]
                    except Exception:
                        pass
                    if fmt_u.startswith("RGB") or fmt_u == "XRGB8888":
                        try:
                            frame_show = cv2.cvtColor(frame_show, cv2.COLOR_RGB2BGR)
                        except Exception:
                            pass
                    cv2.imshow(window_name, frame_show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if window_check_enabled:
                        try:
                            # In some remote / headless / Wayland backends this can
                            # report "not visible" immediately even though display works.
                            # Treat failures as "unsupported" and keep running.
                            v = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                            if v < 0:
                                window_check_enabled = False
                        except Exception:
                            window_check_enabled = False
                    last_display_time = now

            if now >= t_report:
                dt = now - last_report_time
                df = frames - last_report_frames
                fps = (df / dt) if dt > 0 else 0.0
                print(f"FPS (last {dt:.2f}s): {fps:.2f} | frames={frames}")
                last_report_time = now
                last_report_frames = frames
                t_report = now + float(args.report)

        total_dt = time.perf_counter() - t0
        total_fps = (frames / total_dt) if total_dt > 0 else 0.0
        print(f"Total: {frames} frames in {total_dt:.2f}s => {total_fps:.2f} FPS")

    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass

        if args.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
(env) uutzinger@urspi:~/pythonBME210/camera $ python3 testing/picamera2_direct.py --fps 60
Controls set: {'FrameDurationLimits': (16667, 16667)}
=== camera configuration ===
Requested: mode=main size=(640, 480) format=BGR888 fps=60.0 stream_policy=default low_latency=False flip=0
Requested controls: {'FrameDurationLimits': (16667, 16667)}
camera_configuration: {'use_case': 'video', 'transform': <libcamera.Transform 'identity'>, 'colour_space': <libcamera.ColorSpace 'SMPTE170M'>, 'buffer_count': 6, 'queue': True, 'main': {'format': 'BGR888', 'size': (640, 480), 'preserve_ar': True, 'stride': 1920, 'framesize': 921600}, 'lores': None, 'raw': {'format': 'GBRG_PISP_COMP1', 'size': (640, 480), 'stride': 640, 'framesize': 307200}, 'controls': {'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}, 'sensor': {'bit_depth': 10, 'output_size': (640, 480)}, 'display': 'main', 'encode': 'main'}
camera_properties: {'Model': 'ov5647', 'UnitCellSize': (1400, 1400), 'Location': 2, 'Rotation': 0, 'ColorFilterArrangement': 2, 'PixelArraySize': (2592, 1944), 'PixelArrayActiveAreas': [(16, 6, 2592, 1944)], 'ScalerCropMaximum': (16, 0, 2560, 1920), 'SystemDevices': (20752, 20753, 20754, 20755, 20756, 20757, 20758, 20739, 20740, 20741, 20742), 'SensorSensitivity': 1.0}
metadata: FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) AeEnable=None ExposureTime=16836
FPS (last 2.00s): 67.96 | frames=136
FPS (last 2.01s): 67.51 | frames=272
FPS (last 2.00s): 67.49 | frames=407
FPS (last 2.01s): 67.53 | frames=543
Total: 645 frames in 9.56s => 67.48 FPS
"""