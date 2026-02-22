#!/usr/bin/env python3
"""Benchmark binning utilities (square + rectangular).

This compares:
  1) `camera.processor.utilities.bin_rect/bin_square` (Numba or NumPy backend)
  2) pure NumPy reshape+sum reference

Use this to decide whether dedicated specialized kernels are worth keeping.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Iterable

import numpy as np

# Allow running directly from repository checkout without installation.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from camera.processor import utilities as u


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be stack,height,width (e.g. 14,540,720)")
    if min(parts) < 1:
        raise argparse.ArgumentTypeError("shape dimensions must be >= 1")
    return tuple(parts)  # type: ignore[return-value]


def _parse_bins(value: str) -> list[int]:
    vals = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("bins list cannot be empty")
    if min(vals) < 1:
        raise argparse.ArgumentTypeError("bins must be >= 1")
    return vals


def _parse_rect_bins(value: str) -> list[tuple[int, int]]:
    if not value.strip():
        return []
    out: list[tuple[int, int]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" not in token:
            raise argparse.ArgumentTypeError("rect bins must use AxB format, e.g. 2x4,3x5")
        a, b = token.split("x", 1)
        bh = int(a.strip())
        bw = int(b.strip())
        if bh < 1 or bw < 1:
            raise argparse.ArgumentTypeError("rect bins must be >= 1")
        out.append((bh, bw))
    return out


def _np_bin_rect(arr: np.ndarray, bin_h: int, bin_w: int, out_dtype: np.dtype) -> np.ndarray:
    stack, h, w = arr.shape
    h2 = (h // bin_h) * bin_h
    w2 = (w // bin_w) * bin_w
    arr = arr[:, :h2, :w2]
    out_h = h2 // bin_h
    out_w = w2 // bin_w
    out = arr.astype(np.int64, copy=False).reshape(stack, out_h, bin_h, out_w, bin_w).sum(axis=(2, 4), dtype=np.int64)
    if out_dtype == np.uint16:
        np.clip(out, 0, 65535, out=out)
        return out.astype(np.uint16, copy=False)
    np.clip(out, 0, np.iinfo(np.uint32).max, out=out)
    return out.astype(np.uint32, copy=False)


def _bench(fn, loops: int) -> float:
    t0 = time.perf_counter()
    for _ in range(loops):
        fn()
    return time.perf_counter() - t0


def _auto_loops(stack: int, h: int, w: int, target_megapixels: float = 3500.0) -> int:
    pixels = max(1, stack * h * w)
    loops = int(target_megapixels * 1_000_000 / pixels)
    return max(8, loops)


def _report_case(
    arr: np.ndarray,
    bin_h: int,
    bin_w: int,
    loops: int,
    warmup: int,
) -> None:
    stack, h, w = arr.shape
    h2 = (h // bin_h) * bin_h
    w2 = (w // bin_w) * bin_w
    out_h = h2 // bin_h
    out_w = w2 // bin_w

    # Match utilities default dtype policy.
    out_dtype = np.uint16
    if (bin_h == bin_w) and (bin_h in (12, 15, 18, 20)):
        out_dtype = np.uint32
    else:
        max_sum = int(np.iinfo(arr.dtype).max) * bin_h * bin_w
        if max_sum > 65535:
            out_dtype = np.uint32

    out_fast = np.empty((stack, out_h, out_w), dtype=out_dtype)
    out_np = np.empty((stack, out_h, out_w), dtype=out_dtype)

    def run_fast():
        return u.bin_rect(arr, bin_h=bin_h, bin_w=bin_w, out_dtype=out_dtype, out=out_fast)

    def run_np():
        np.copyto(out_np, _np_bin_rect(arr, bin_h=bin_h, bin_w=bin_w, out_dtype=out_dtype), casting="unsafe")
        return out_np

    for _ in range(warmup):
        run_fast()
        run_np()

    t_fast = _bench(run_fast, loops)
    t_np = _bench(run_np, loops)

    # one result diff check
    run_fast()
    run_np()
    diff = out_fast.astype(np.int64) - out_np.astype(np.int64)
    max_abs = int(np.max(np.abs(diff))) if diff.size else 0

    in_mpix = (stack * h2 * w2) / 1_000_000.0
    fast_ops = loops / t_fast
    np_ops = loops / t_np
    fast_mpix = in_mpix * fast_ops
    np_mpix = in_mpix * np_ops
    speedup = t_np / t_fast if t_fast > 0.0 else 0.0

    print(
        f"bin={bin_h}x{bin_w: <2d} "
        f"| out={out_dtype.__name__:<6s} "
        f"| fast={fast_ops:8.1f} ops/s ({fast_mpix:8.1f} MPix/s) "
        f"| numpy={np_ops:8.1f} ops/s ({np_mpix:8.1f} MPix/s) "
        f"| speedup={speedup:5.2f}x | max_abs={max_abs}"
    )


def _iter_square_bins(vals: Iterable[int]) -> Iterable[tuple[int, int]]:
    for b in vals:
        yield b, b


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark processor binning utilities.")
    ap.add_argument("--shape", type=_parse_shape, default="14,540,720", help="Input shape stack,height,width.")
    ap.add_argument("--bins", type=_parse_bins, default="2,3,4,5,6,8,9,10,12,15,18,20", help="Square bins.")
    ap.add_argument("--rect-bins", type=_parse_rect_bins, default="2x4,3x5", help="Rectangular bins AxB.")
    ap.add_argument("--dtype", choices=["uint8", "uint16"], default="uint8", help="Input dtype.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--loops", type=int, default=0, help="Loops per case (0=auto).")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup runs per case.")
    args = ap.parse_args()

    stack, h, w = args.shape
    dtype = np.uint8 if args.dtype == "uint8" else np.uint16
    rng = np.random.default_rng(args.seed)

    max_val = 256 if dtype == np.uint8 else 65536
    arr = rng.integers(0, max_val, size=(stack, h, w), dtype=dtype)

    loops = args.loops if args.loops > 0 else _auto_loops(stack, h, w)

    print("Binning benchmark")
    print(f"  shape            : {(stack, h, w)}")
    print(f"  dtype            : {dtype.__name__}")
    print(f"  backend          : {'numba' if u._NUMBA_AVAILABLE else 'numpy'}")
    print(f"  loops per case   : {loops}")
    print(f"  warmup           : {args.warmup}")
    print("")

    for bh, bw in _iter_square_bins(args.bins):
        _report_case(arr, bh, bw, loops=loops, warmup=args.warmup)

    if args.rect_bins:
        print("")
        for bh, bw in args.rect_bins:
            _report_case(arr, bh, bw, loops=loops, warmup=args.warmup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

