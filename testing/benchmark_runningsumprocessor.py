#!/usr/bin/env python3
"""Benchmark running-sum highpass against one-pole highpass processors.

Example:
    python3 testing/benchmark_runningsumprocessor.py
    python3 testing/benchmark_runningsumprocessor.py --shape 14,540,720 --delay 16 --frames 200
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

# Allow running directly from repository checkout without installation.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from camera.processor.highpassprocessor import (
    highpassProcessor as onepoleHighpassProcessor,
    one_pole_alpha_from_fc,
    one_pole_backend,
    one_pole_fc_from_alpha,
)
from camera.processor import runningsumprocessor as _runsum_mod
from camera.processor.runningsumprocessor import (
    highpassProcessor as runsumHighpassProcessor,
    running_sum_delay_to_fc,
    running_sum_fc_to_delay,
)

def _parse_shape(value: str) -> tuple[int, ...]:
    parts = [int(p.strip()) for p in value.split(",") if p.strip()]
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("shape must have at least two dimensions, e.g. 540,720 or 14,540,720")
    return tuple(parts)

def _run_runningsum(
    cubes: np.ndarray,
    shape: tuple[int, ...],
    delay: int,
    warmup: int,
) -> dict:
    proc = runsumHighpassProcessor(res=shape, delay=delay)

    for i in range(warmup):
        np.copyto(proc._x, cubes[i % cubes.shape[0]], casting="unsafe")
        proc._process_core()

    t0 = time.perf_counter()
    for i in range(cubes.shape[0]):
        np.copyto(proc._x, cubes[i], casting="unsafe")
        proc._process_core()
    elapsed = time.perf_counter() - t0

    out = proc.data_highpass.copy()
    cps = cubes.shape[0] / elapsed if elapsed > 0.0 else 0.0
    ms = 1000.0 * elapsed / cubes.shape[0] if cubes.shape[0] > 0 else 0.0

    return {
        "impl": "numba" if getattr(_runsum_mod, "_NUMBA_AVAILABLE", False) else "numpy",
        "elapsed_s": elapsed,
        "cps": cps,
        "ms_per_cube": ms,
        "out": out,
    }

def _run_onepole_processor(
    cubes: np.ndarray,
    shape: tuple[int, ...],
    alpha: float,
    warmup: int,
) -> dict:
    proc = onepoleHighpassProcessor(res=shape, alpha=alpha)

    for i in range(warmup):
        np.copyto(proc._x, cubes[i % cubes.shape[0]], casting="unsafe")
        proc._process_core()

    t0 = time.perf_counter()
    for i in range(cubes.shape[0]):
        np.copyto(proc._x, cubes[i], casting="unsafe")
        proc._process_core()
    elapsed = time.perf_counter() - t0

    out = proc.filteredData.copy()
    cps = cubes.shape[0] / elapsed if elapsed > 0.0 else 0.0
    ms = 1000.0 * elapsed / cubes.shape[0] if cubes.shape[0] > 0 else 0.0
    return {
        "impl": one_pole_backend(),
        "elapsed_s": elapsed,
        "cps": cps,
        "ms_per_cube": ms,
        "out": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark running-sum highpass and one-pole highpass processors.")
    ap.add_argument("--shape", type=_parse_shape, default="14,580,740", help="Frame/cube shape, comma-separated.")
    ap.add_argument("--delay", type=int, default=8, help="Running-sum delay/window length D.")
    ap.add_argument("--frames", type=int, default=200, help="Number of cubes to process.")
    ap.add_argument("--warmup", type=int, default=4, help="Warmup iterations.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--fs", type=float, default=30.0, help="Sample rate for cutoff guidance output.")
    ap.add_argument("--onepole-fc", type=float, default=1.0, help="One-pole -3dB frequency [Hz].")
    args = ap.parse_args()

    shape = args.shape
    delay = int(args.delay)
    n_cubes = int(args.frames)

    if delay < 1:
        raise SystemExit("delay must be >= 1")
    if n_cubes < 1:
        raise SystemExit("cubes must be >= 1")

    rng = np.random.default_rng(args.seed)
    cubes = rng.integers(low=0, high=256, size=(n_cubes, *shape), dtype=np.uint8)

    print("Benchmark settings")
    print(f"  shape          : {shape}")
    print(f"  delay (D)      : {delay}")
    print(f"  cubes          : {n_cubes}")
    print(f"  warmup         : {args.warmup}")
    print(f"  fs [Hz]        : {args.fs}")
    print(f"  runsum fc [Hz] : {running_sum_delay_to_fc(args.fs, delay):.6f} (approx -3 dB)")
    print(f"  runsum D from fc={args.onepole_fc} Hz : {running_sum_fc_to_delay(args.fs, args.onepole_fc)}")
    print(f"  one-pole impl  : {one_pole_backend()}")

    alpha = one_pole_alpha_from_fc(args.fs, args.onepole_fc)
    fc_back = one_pole_fc_from_alpha(args.fs, alpha)
    print(f"  one-pole alpha : {alpha:.8f} (from fc={args.onepole_fc} Hz, inverse fc={fc_back:.8f} Hz)")
    print("")

    runsum_res = _run_runningsum(cubes, shape, delay, args.warmup)
    onepole_res = _run_onepole_processor(cubes, shape, alpha, args.warmup)

    print(
        f"{'runsum':>8s} -> {runsum_res['impl']:<6s} | "
        f"{runsum_res['cps']:9.2f} cubes/s | {runsum_res['ms_per_cube']:8.3f} ms/cube"
    )
    print(
        f"{'onepole':>8s} -> {onepole_res['impl']:<6s} | "
        f"{onepole_res['cps']:9.2f} cubes/s | {onepole_res['ms_per_cube']:8.3f} ms/cube"
    )

    diff = onepole_res["out"] - runsum_res["out"]
    rms = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    print(f"delta(onepole vs runsum): rms={rms:.6e}, max_abs={max_abs:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
