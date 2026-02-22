###############################################################################
# One-pole lowpass/highpass processor
#
# This module implements a one-pole lowpass and derived highpass filter
# 
# Urs Utzinger 2022-26
# Gpt 5.2 2026
###############################################################################
#
# Lowpass recurrence ("poor man's lowpass"):
#   y_lp[n] = (1 - alpha) * y_lp[n-1] + alpha * x[n]
#
# Highpass is derived from the same state:
#   y_hp[n] = x[n] - y_lp[n]
#
# Alpha selection from cutoff (exact -3 dB mapping):
#
#   fs: smapling frequency 
#   fc: desired -3 dB cutoff frequency
#       wc = 2 * pi * fc / fs
#       t = 1 - cos(2*pi*fc/fs)
#       alpha = -t + sqrt(t*t + 2*t)
###############################################################################

from __future__ import annotations

import logging
import time
from math import acos, cos, pi, sqrt
from queue import Empty, Queue
from threading import Thread
from typing import Optional

import numpy as np

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False

def alpha_from_fc(f_s: float = 50.0, f_c: float = 5.0) -> float:
    """Alpha for y=(1-alpha)*y_prev + alpha*x with -3 dB at f_c."""
    if f_s <= 0.0:
        raise ValueError("f_s must be > 0")
    if f_c <= 0.0 or f_c >= (0.5 * f_s):
        raise ValueError("f_c must satisfy 0 < f_c < f_s/2")
    w_c = (2.0 * pi) * f_c / f_s
    t = 1.0 - cos(w_c)
    return float(-t + sqrt((t * t) + (2.0 * t)))

def fc_from_alpha(f_s: float, alpha: float) -> float:
    """Inverse mapping from alpha to one-pole -3 dB cutoff."""
    if f_s <= 0.0:
        raise ValueError("f_s must be > 0")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    cos_w = 1.0 - (alpha * alpha) / (2.0 * (1.0 - alpha))
    cos_w = min(1.0, max(-1.0, cos_w))
    return float(f_s * (acos(cos_w) / (2.0 * pi)))

def one_pole_alpha_from_fc(f_s: float = 50.0, f_c: float = 5.0) -> float:
    """Compatibility alias for alpha_from_fc."""
    return alpha_from_fc(f_s=f_s, f_c=f_c)

def one_pole_fc_from_alpha(f_s: float, alpha: float) -> float:
    """Compatibility alias for fc_from_alpha."""
    return fc_from_alpha(f_s=f_s, alpha=alpha)

# Backward compatibility with older examples.
computeAlpha = alpha_from_fc

def lp_coefficients(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference scipy.signal.lfilter coefficients for one-pole lowpass."""
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    b = np.array([float(alpha)], dtype=np.float64)
    a = np.array([1.0, -(1.0 - float(alpha))], dtype=np.float64)
    return b, a

def hp_coefficients(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference scipy.signal.lfilter coefficients for derived one-pole highpass."""
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    beta = 1.0 - float(alpha)
    b = np.array([beta, -beta], dtype=np.float64)
    a = np.array([1.0, -beta], dtype=np.float64)
    return b, a

def one_pole_lp_coefficients(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility alias for lp_coefficients."""
    return lp_coefficients(alpha)

def one_pole_hp_coefficients(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility alias for hp_coefficients."""
    return hp_coefficients(alpha)

def backend() -> str:
    """Return active one-pole compute backend."""
    return "numba" if _NUMBA_AVAILABLE else "numpy"

def one_pole_backend() -> str:
    """Compatibility alias for backend."""
    return backend()

if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def one_pole_step_lp_hp_numba(
        x_flat: np.ndarray,
        lp_flat: np.ndarray,
        hp_flat: np.ndarray,
        alpha: np.float32,
    ) -> None:
        one_minus_a = np.float32(1.0 - alpha)
        n = x_flat.size
        for i in prange(n):
            lp_i = (one_minus_a * lp_flat[i]) + (alpha * x_flat[i])
            lp_flat[i] = lp_i
            hp_flat[i] = x_flat[i] - lp_i

    @njit(cache=True, fastmath=True, parallel=True)
    def one_pole_step_lp_numba(
        x_flat: np.ndarray,
        lp_flat: np.ndarray,
        alpha: np.float32,
    ) -> None:
        one_minus_a = np.float32(1.0 - alpha)
        n = x_flat.size
        for i in prange(n):
            lp_flat[i] = (one_minus_a * lp_flat[i]) + (alpha * x_flat[i])

def _apply_one_pole_step_lp_hp(
    x_f32: np.ndarray,
    lp: np.ndarray,
    hp: np.ndarray,
    alpha: np.float32,
) -> None:
    """Update lowpass/highpass in place from one input array."""
    if _NUMBA_AVAILABLE:
        one_pole_step_lp_hp_numba(x_f32.reshape(-1), lp.reshape(-1), hp.reshape(-1), alpha)
        return

    # Pure NumPy in-place recurrence.
    one_minus_a = np.float32(1.0 - float(alpha))
    np.multiply(lp, one_minus_a, out=lp)
    lp += alpha * x_f32
    np.subtract(x_f32, lp, out=hp)

def _apply_one_pole_step_lp(x_f32: np.ndarray, lp: np.ndarray, alpha: np.float32) -> None:
    """Update lowpass only in place from one input array."""
    if _NUMBA_AVAILABLE:
        one_pole_step_lp_numba(x_f32.reshape(-1), lp.reshape(-1), alpha)
        return

    one_minus_a = np.float32(1.0 - float(alpha))
    np.multiply(lp, one_minus_a, out=lp)
    lp += alpha * x_f32

def one_pole_step_inplace(x_f32: np.ndarray, lp: np.ndarray, hp: np.ndarray, alpha: float) -> None:
    """Public helper for one-pole LP/HP update used by benchmarks and tests."""
    if x_f32.shape != lp.shape or x_f32.shape != hp.shape:
        raise ValueError("x_f32, lp, and hp must have identical shapes")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if x_f32.dtype != np.float32:
        raise ValueError("x_f32 must be float32")
    if lp.dtype != np.float32 or hp.dtype != np.float32:
        raise ValueError("lp and hp must be float32")
    _apply_one_pole_step_lp_hp(x_f32, lp, hp, np.float32(alpha))

def one_pole_step_lp_inplace(x_f32: np.ndarray, lp: np.ndarray, alpha: float) -> None:
    """Public helper for one-pole lowpass-only update."""
    if x_f32.shape != lp.shape:
        raise ValueError("x_f32 and lp must have identical shapes")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if x_f32.dtype != np.float32 or lp.dtype != np.float32:
        raise ValueError("x_f32 and lp must be float32")
    _apply_one_pole_step_lp(x_f32, lp, np.float32(alpha))

class highpassProcessor(Thread):
    """One-pole lowpass + highpass processor.

    Input queue:
      - input: (data_time, data)

    Output queues:
      - output_highpass: (data_time, highpass)
      - output_lowpass:  (data_time, lowpass)
      - output is an alias to output_highpass for consistency with runningsum.
    """

    def __init__(self, res: tuple[int, ...], alpha: float = 0.95, queue_size: int = 32):
        self.stopped = True

        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        self.alpha = np.float32(alpha)
        self._res = tuple(int(v) for v in res)

        self.averageData = np.zeros(self._res, dtype=np.float32)   # lowpass
        self.filteredData = np.zeros(self._res, dtype=np.float32)  # highpass
        self._x = np.zeros(self._res, dtype=np.float32)

        self.input = Queue(maxsize=queue_size)
        self.output_highpass = Queue(maxsize=queue_size)
        self.output_lowpass = Queue(maxsize=queue_size)
        self.output = self.output_highpass
        self.log = Queue(maxsize=32)
        self.stopped = True

        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Optional[Thread] = None

        Thread.__init__(self)

        impl = "numba" if _NUMBA_AVAILABLE else "numpy"
        if not self.log.full():
            self.log.put_nowait((logging.INFO, f"OnePole:impl={impl}, alpha={float(self.alpha):.6f}"))

    def stop(self):
        """Stop the processor thread."""
        self.stopped = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def start(self):
        """Start the processor thread."""
        if not self.stopped:
            return
        self.stopped = False
        self._thread = Thread(target=self.update)
        self._thread.daemon = True
        self._thread.start()

    def _process_core(self):
        _apply_one_pole_step_lp_hp(self._x, self.averageData, self.filteredData, self.alpha)

    def update(self):
        """Thread processing loop."""
        last_time = time.time()
        num_cubes = 0
        total_time = 0.0

        while not self.stopped:
            try:
                data_time, data = self.input.get(block=True, timeout=0.25)
            except Empty:
                continue

            start_time = time.perf_counter()
            np.copyto(self._x, data, casting="unsafe")
            self._process_core()
            total_time += time.perf_counter() - start_time

            if not self.output_highpass.full():
                self.output_highpass.put_nowait((data_time, self.filteredData))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "OnePoleHP:output_highpass queue is full"))

            if not self.output_lowpass.full():
                self.output_lowpass.put_nowait((data_time, self.averageData))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "OnePoleHP:output_lowpass queue is full"))

            num_cubes += 1
            current_time = time.time()
            if (current_time - last_time) >= 5.0:
                self.measured_cps = num_cubes / 5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"HighPass:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"HighPass:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_time = current_time

class lowpassProcessor(Thread):
    """One-pole lowpass-only processor.

    Input queue:
      - input: (data_time, data)

    Output queue:
      - output: (data_time, lowpass)
    """

    def __init__(self, res: tuple[int, ...], alpha: float = 0.95, queue_size: int = 32):
        self.stopped = True

        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        self.alpha = np.float32(alpha)
        self._res = tuple(int(v) for v in res)

        self.averageData = np.zeros(self._res, dtype=np.float32)
        self._x = np.zeros(self._res, dtype=np.float32)

        self.input = Queue(maxsize=queue_size)
        self.output = Queue(maxsize=queue_size)
        self.log = Queue(maxsize=32)
        self.stopped = True

        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Optional[Thread] = None

        Thread.__init__(self)

        impl = "numba" if _NUMBA_AVAILABLE else "numpy"
        if not self.log.full():
            self.log.put_nowait((logging.INFO, f"OnePoleLP:impl={impl}, alpha={float(self.alpha):.6f}"))

    def stop(self):
        self.stopped = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def start(self):
        if not self.stopped:
            return
        self.stopped = False
        self._thread = Thread(target=self.update)
        self._thread.daemon = True
        self._thread.start()

    def _process_core(self):
        _apply_one_pole_step_lp(self._x, self.averageData, self.alpha)

    def update(self):
        last_time = time.time()
        num_cubes = 0
        total_time = 0.0

        while not self.stopped:
            try:
                data_time, data = self.input.get(block=True, timeout=0.25)
            except Empty:
                continue

            start_time = time.perf_counter()
            np.copyto(self._x, data, casting="unsafe")
            self._process_core()
            total_time += time.perf_counter() - start_time

            if not self.output.full():
                self.output.put_nowait((data_time, self.averageData))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "OnePoleLP:output queue is full"))

            num_cubes += 1
            current_time = time.time()
            if (current_time - last_time) >= 5.0:
                self.measured_cps = num_cubes / 5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"LowPass:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"LowPass:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_time = current_time
