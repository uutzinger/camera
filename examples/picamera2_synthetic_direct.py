#!/usr/bin/env python3
"""Direct synthetic FPS probe (no Picamera2/libcamera).

This mirrors examples/picamera2_direct.py, but uses PiCamera2Core in synthetic
mode (configs['test_pattern']) so we can measure loop + numpy work without
camera I/O.

Run:
  python3 examples/picamera2_synthetic_direct.py --fps 0
  python3 examples/picamera2_synthetic_direct.py --fps 1000 --no-display

Notes:
- `--fps 0` means unpaced (as fast as possible).
- This is intended to help compare:
    direct-Picamera2 vs wrapper-Picamera2 vs wrapper-synthetic
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Direct synthetic FPS probe (PiCamera2Core, no camera)")
    ap.add_argument("--size", type=_parse_size, default=(640, 480), help="Frame size WxH")
    ap.add_argument(
        "--format",
        type=str,
        default="BGR888",
        help="Synthetic main format (e.g. BGR888, RGB888, XRGB8888, XBGR8888, YUV420, YUYV)",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="gradient",
        help="Pattern: gradient|checker|noise|static",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Requested synthetic FPS. 0 = unpaced.",
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
        help="Max display refresh rate (Hz). 0 = update every generated frame.",
    )
    ap.add_argument("--duration", type=float, default=15.0, help="Total run time in seconds")
    ap.add_argument("--warmup", type=float, default=1.0, help="Warmup time in seconds (not counted)")
    ap.add_argument("--report", type=float, default=2.0, help="Report interval in seconds")
    args = ap.parse_args()

    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")

    cv2 = None
    if args.display:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            print(f"WARNING: OpenCV (cv2) not available; disabling display: {exc}")
            args.display = False

    from queue import Queue
    from camera.capture.picamera2core import PiCamera2Core

    log_q: Queue = Queue(maxsize=64)
    configs = {
        "mode": "main",
        "camera_res": args.size,
        "format": args.format,
        "fps": float(args.fps),
        "test_pattern": args.pattern,
        "flip": 0,
        "output_res": (-1, -1),
        "low_latency": True,
    }

    core = PiCamera2Core(configs, camera_num=0, log_queue=log_q)
    if not core.open_cam():
        print("ERROR: failed to open synthetic core")
        while not log_q.empty():
            _lvl, msg = log_q.get_nowait()
            print(msg)
        return 2

    while not log_q.empty():
        _lvl, msg = log_q.get_nowait()
        print(msg)

    window_name = "Synthetic direct"
    display_interval = 0.0
    if args.display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)
        except Exception as exc:
            print(f"WARNING: OpenCV window init failed; disabling display: {exc}")
            args.display = False

        try:
            dfps = float(args.display_fps or 0.0)
            display_interval = 0.0 if dfps <= 0 else (1.0 / dfps)
        except Exception:
            display_interval = 0.0

    warmup_end = time.perf_counter() + float(args.warmup)
    while time.perf_counter() < warmup_end:
        _ = core.capture_array()

    t0 = time.perf_counter()
    t_end = t0 + float(args.duration)
    t_report = t0 + float(args.report)

    last_display_time = 0.0
    frames = 0
    last_report_frames = 0
    last_report_time = t0

    try:
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            frame, _ts = core.capture_array()
            frames += 1

            if args.display and frame is not None:
                if display_interval <= 0.0 or (now - last_display_time) >= display_interval:
                    frame_show = frame
                    # Convert to OpenCV BGR for display if needed.
                    fmt_u = str(args.format or "").upper()
                    try:
                        if fmt_u in ("RGB888", "XBGR8888"):
                            # RGB/RGBA -> BGR
                            frame_show = cv2.cvtColor(frame_show, cv2.COLOR_RGB2BGR)
                        elif fmt_u == "XRGB8888":
                            # BGRA -> BGR
                            frame_show = cv2.cvtColor(frame_show, cv2.COLOR_BGRA2BGR)
                        elif fmt_u == "YUV420":
                            frame_show = cv2.cvtColor(frame_show, cv2.COLOR_YUV2BGR_I420)
                        elif fmt_u == "YUYV":
                            frame_show = cv2.cvtColor(frame_show, cv2.COLOR_YUV2BGR_YUY2)
                        else:
                            # BGR888 and unknown fallbacks
                            pass
                    except Exception:
                        pass

                    cv2.imshow(window_name, frame_show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
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
            core.close_cam()
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
