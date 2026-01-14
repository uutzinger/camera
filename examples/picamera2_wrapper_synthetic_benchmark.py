#!/usr/bin/env python3
"""Synthetic wrapper benchmark (no Picamera2/libcamera).

This exercises:
- the non-Qt wrapper capture thread (`piCamera2Capture`)
- the SPSC `FrameBuffer` push/pull/copy behavior

It does NOT touch Picamera2/libcamera. Frames are generated in `PiCamera2Core`
when `configs['test_pattern']` is enabled.

Examples:
  python3 examples/picamera2_wrapper_synthetic_benchmark.py
  python3 examples/picamera2_wrapper_synthetic_benchmark.py --fps 120 --size 640x480
  python3 examples/picamera2_wrapper_synthetic_benchmark.py --fps 0 --consume latest
  python3 examples/picamera2_wrapper_synthetic_benchmark.py --fps 240 --consume copy
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(description="Benchmark wrapper + FrameBuffer using synthetic frames")
    ap.add_argument("--size", type=_parse_size, default=(640, 480), help="Synthetic frame size WxH")
    ap.add_argument(
        "--pattern",
        type=str,
        default="gradient",
        help="Synthetic pattern: gradient|checker|noise|static",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Synthetic generator FPS. 0 = unpaced (as fast as possible).",
    )
    ap.add_argument(
        "--consume",
        choices=("none", "latest", "copy"),
        default="latest",
        help=(
            "Consumer behavior: none (do not pull), latest (drain with copy=False), "
            "copy (pull one and copy it)."
        ),
    )
    ap.add_argument("--duration", type=float, default=10.0, help="Run time in seconds")
    ap.add_argument("--report", type=float, default=2.0, help="Report interval in seconds")
    args = ap.parse_args()

    from camera.capture.picamera2capture import piCamera2Capture

    configs = {
        "mode": "main",
        "camera_res": args.size,
        "output_res": (-1, -1),
        "flip": 0,
        "low_latency": True,
        "buffersize": 4,
        "fps": float(args.fps),
        "test_pattern": args.pattern,
        "format": "BGR3",
        "stream_policy": "default",
    }

    cam = piCamera2Capture(configs, camera_num=0)
    if not cam.open_cam():
        print("Failed to open synthetic wrapper")
        return 2

    # Start producing
    cam.start()

    t0 = time.perf_counter()
    t_end = t0 + float(args.duration)
    t_report = t0 + float(args.report)

    cons_frames = 0
    cons_last_frames = 0
    cons_last_t = t0

    try:
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break

            buf = getattr(cam, "buffer", None)
            if args.consume != "none" and buf is not None:
                if args.consume == "latest":
                    last = None
                    try:
                        while buf.avail() > 0:
                            last = buf.pull(copy=False)
                    except Exception:
                        last = None
                    if last and last[0] is not None:
                        cons_frames += 1
                else:  # copy
                    if buf.avail() > 0:
                        frame, _ts = buf.pull(copy=False)
                        if frame is not None:
                            try:
                                _ = frame.copy()
                            except Exception:
                                pass
                            cons_frames += 1

            # Drain log queue (optional visibility)
            try:
                while not cam.log.empty():
                    level, msg = cam.log.get_nowait()
                    # Keep output minimal.
                    if "Open summary" in str(msg) or "Synthetic" in str(msg):
                        print(msg)
            except Exception:
                pass

            if now >= t_report:
                # Producer fps (wrapper measured)
                prod_fps = float(getattr(cam, "measured_fps", 0.0) or 0.0)
                # Consumer fps (how often we successfully updated our consumer)
                dt = now - cons_last_t
                df = cons_frames - cons_last_frames
                cons_fps = (df / dt) if dt > 0 else 0.0
                print(f"Producer FPS (wrapper): {prod_fps:.1f} | Consumer FPS: {cons_fps:.1f} | consume={args.consume}")
                cons_last_t = now
                cons_last_frames = cons_frames
                t_report = now + float(args.report)

            # Don’t busy-spin.
            time.sleep(0.0005)

    finally:
        try:
            cam.stop()
            cam.join(timeout=2.0)
            cam.close_cam()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
