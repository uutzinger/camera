###############################################################################
# Three-band processor (low / mid / high)
#
# This module implements a three-band filter with equalizer option.
# It cascades four one-pole lowpass filters together
# 
# Urs Utzinger 2022-26
# Gpt 5.2 2026
###############################################################################
#
# Single cascaded one-pole lowpass:
#   y[n] = y[n-1] + a(x[n] - y[n-1])
# Transfer function in z-domain:
#   H(z) = a / (1 - (1-a) z^-1)
#   pole at z = 1 - a (inside unit circle for 0<a<1)
#
# 4 cascaded to gether:
#   H(z) = [ a / (1 - (1-a) z^-1 ) ]^4
#
# Bands are derived from two cascaded 4-pole lowpass branches:
#   low      = LP4(fc_low, x)
#   mid      = LP4(fc_high, x) - LP4(fc_low, x)
#   high     = x - LP4(fc_high, x)
#   equalized = gl*low + gm*mid + gh*high
###############################################################################

from __future__ import annotations

import logging
import time
from math import cos, pi, sqrt
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


def alpha_from_fc(
    fs: float,
    fc: float,
    cascades: int = 1,
    *
) -> float:
    """Map cutoff to alpha for one-pole or cascaded equal-pole lowpass.

    For cascades=1 this matches the exact one-pole -3 dB mapping used elsewhere.
    For cascades>1, alpha is chosen so the full cascade has -3 dB at fc.
    `poles` is a backward-compatible alias for `cascades`.
    """

    if fs <= 0.0:
        raise ValueError("fs must be > 0")
    if fc <= 0.0 or fc >= 0.5 * fs:
        raise ValueError("fc must satisfy 0 < fc < fs/2")
    if cascades < 1:
        raise ValueError("cascades must be >= 1")

    w_c = (2.0 * pi) * fc / fs
    t = 1.0 - cos(w_c)

    if cascades == 1:
        return float(-t + sqrt((t * t) + (2.0 * t)))

    # Target |H|^2 for each pole in the cascade at wc:
    # |H_total|^2 = |H|^(2*cascades) = 1/2  ->  |H|^2 = 2^(-1/cascades)
    p = float(2.0 ** (-1.0 / float(cascades)))
    a = (1.0 - p)
    b =  2.0 * p * t
    c = -2.0 * p * t
    disc = (b * b) - (4.0 * a * c)
    alpha = (-b + sqrt(max(disc, 0.0))) / (2.0 * a)
    if alpha <= 0.0:
        alpha = 1e-6
    if alpha >= 1.0:
        alpha = 1.0 - 1e-6
    return float(alpha)

def backend() -> str:
    return "numba" if _NUMBA_AVAILABLE else "numpy"


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _threeband_step_numba(
        x_flat: np.ndarray,
        l0:     np.ndarray,
        l1:     np.ndarray,
        l2:     np.ndarray,
        l3:     np.ndarray,
        h0:     np.ndarray,
        h1:     np.ndarray,
        h2:     np.ndarray,
        h3:     np.ndarray,
        low:    np.ndarray,
        mid:    np.ndarray,
        high:   np.ndarray,
        out:    np.ndarray,
        alpha_low:  np.float32,
        alpha_high: np.float32,
        gain_low:   np.float32,
        gain_mid:   np.float32,
        gain_high:  np.float32,
    ) -> None:
        n = x_flat.size
        for i in prange(n):
            x_i = x_flat[i]

            # 4-pole LP at fc_low
            s0 = l0[i] + alpha_low * (x_i - l0[i])
            l0[i] = s0
            s1 = l1[i] + alpha_low * (s0 - l1[i])
            l1[i] = s1
            s2 = l2[i] + alpha_low * (s1 - l2[i])
            l2[i] = s2
            s3 = l3[i] + alpha_low * (s2 - l3[i])
            l3[i] = s3
            low_i = s3

            # 4-pole LP at fc_high
            t0 = h0[i] + alpha_high * (x_i - h0[i]); h0[i] = t0
            t1 = h1[i] + alpha_high * (t0 - h1[i]);  h1[i] = t1
            t2 = h2[i] + alpha_high * (t1 - h2[i]);  h2[i] = t2
            t3 = h3[i] + alpha_high * (t2 - h3[i]);  h3[i] = t3

            mid_i = t3 - low_i
            high_i = x_i - t3

            low[i] = low_i
            mid[i] = mid_i
            high[i] = high_i
            out[i] = (gain_low * low_i) + (gain_mid * mid_i) + (gain_high * high_i)

def _threeband_step_numpy(
    x:    np.ndarray,
    l0:   np.ndarray,
    l1:   np.ndarray,
    l2:   np.ndarray,
    l3:   np.ndarray,
    h0:   np.ndarray,
    h1:   np.ndarray,
    h2:   np.ndarray,
    h3:   np.ndarray,
    low:  np.ndarray,
    mid:  np.ndarray,
    high: np.ndarray,
    out:  np.ndarray,
    alpha_low:  np.float32,
    alpha_high: np.float32,
    gain_low:   np.float32,
    gain_mid:   np.float32,
    gain_high:  np.float32,
) -> None:

    one_minus_low =  np.float32(1.0 - float(alpha_low))
    one_minus_high = np.float32(1.0 - float(alpha_high))

    np.multiply(l0, one_minus_low,  out=l0); l0 += alpha_low * x
    np.multiply(l1, one_minus_low,  out=l1); l1 += alpha_low * l0
    np.multiply(l2, one_minus_low,  out=l2); l2 += alpha_low * l1
    np.multiply(l3, one_minus_low,  out=l3); l3 += alpha_low * l2
    np.copyto(low, l3)

    np.multiply(h0, one_minus_high, out=h0); h0 += alpha_high * x
    np.multiply(h1, one_minus_high, out=h1); h1 += alpha_high * h0
    np.multiply(h2, one_minus_high, out=h2); h2 += alpha_high * h1
    np.multiply(h3, one_minus_high, out=h3); h3 += alpha_high * h2

    np.subtract(h3, low, out=mid)
    np.subtract(x,  h3,  out=high)

    np.multiply(low, gain_low, out=out)
    out += gain_mid * mid
    out += gain_high * high

class threebandProcessor(Thread):
    """Threaded low/mid/high splitter with optional equalized output."""

    def __init__(
        self,
        res: tuple[int, ...],
        fs: float,
        fc_low: float,
        fc_high: float,
        gain_low: float = 1.0,
        gain_mid: float = 1.0,
        gain_high: float = 1.0,
        cascades: int = 4,
        queue_size: int = 32,
    ):
        self.stopped = True

        if fc_low <= 0.0 or fc_high <= 0.0:
            raise ValueError("fc_low and fc_high must be > 0")
        if fc_low >= fc_high:
            raise ValueError("fc_low must be < fc_high")
        if fc_high >= 0.5 * fs:
            raise ValueError("fc_high must be < fs/2")

        if cascades != 4:
            raise ValueError("threebandProcessor currently supports cascades=4 only")

        self._res = tuple(int(v) for v in res)
        self.fs = float(fs)
        self.fc_low = float(fc_low)
        self.fc_high = float(fc_high)

        self.alpha_low = np.float32(alpha_from_fc(self.fs, self.fc_low, cascades=4))
        self.alpha_high = np.float32(alpha_from_fc(self.fs, self.fc_high, cascades=4))

        self.gain_low = np.float32(gain_low)
        self.gain_mid = np.float32(gain_mid)
        self.gain_high = np.float32(gain_high)

        # Low-cut LP4 states
        self._l0 = np.zeros(self._res, dtype=np.float32)
        self._l1 = np.zeros(self._res, dtype=np.float32)
        self._l2 = np.zeros(self._res, dtype=np.float32)
        self._l3 = np.zeros(self._res, dtype=np.float32)

        # High-cut LP4 states
        self._h0 = np.zeros(self._res, dtype=np.float32)
        self._h1 = np.zeros(self._res, dtype=np.float32)
        self._h2 = np.zeros(self._res, dtype=np.float32)
        self._h3 = np.zeros(self._res, dtype=np.float32)

        # Inpput
        self._x = np.zeros(self._res, dtype=np.float32)
        # Outputs
        self.data_low = np.zeros(self._res, dtype=np.float32)
        self.data_mid = np.zeros(self._res, dtype=np.float32)
        self.data_high = np.zeros(self._res, dtype=np.float32)
        self.data_equalized = np.zeros(self._res, dtype=np.float32)

        self.input            = Queue(maxsize=queue_size)
        self.output_low       = Queue(maxsize=queue_size)
        self.output_mid       = Queue(maxsize=queue_size)
        self.output_high      = Queue(maxsize=queue_size)
        self.output_equalized = Queue(maxsize=queue_size)  # equalized
        self.output           = self.output_equalized      # backward-compatible alias
        self.log = Queue(maxsize=32)
        self.stopped = True

        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Optional[Thread] = None

        Thread.__init__(self)

        impl = backend()
        if not self.log.full():
            self.log.put_nowait(
                (
                    logging.INFO,
                    (
                        f"ThreeBand:impl={impl}, cascades=4, fs={self.fs:.3f}, "
                        f"fc_low={self.fc_low:.3f}, fc_high={self.fc_high:.3f}, "
                        f"alpha_low={float(self.alpha_low):.6f}, alpha_high={float(self.alpha_high):.6f}"
                    ),
                )
            )

    def set_gains(self, gain_low: float, gain_mid: float, gain_high: float) -> None:
        self.gain_low  = np.float32(gain_low)
        self.gain_mid  = np.float32(gain_mid)
        self.gain_high = np.float32(gain_high)

    def reset(self) -> None:
        self._l0.fill(0.0)
        self._l1.fill(0.0)
        self._l2.fill(0.0)
        self._l3.fill(0.0)
        self._h0.fill(0.0)
        self._h1.fill(0.0)
        self._h2.fill(0.0)
        self._h3.fill(0.0)

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
        if _NUMBA_AVAILABLE:
            _threeband_step_numba(
                self._x.reshape(-1),
                self._l0.reshape(-1),
                self._l1.reshape(-1),
                self._l2.reshape(-1),
                self._l3.reshape(-1),
                self._h0.reshape(-1),
                self._h1.reshape(-1),
                self._h2.reshape(-1),
                self._h3.reshape(-1),
                self.data_low.reshape(-1),
                self.data_mid.reshape(-1),
                self.data_high.reshape(-1),
                self.data_equalized.reshape(-1),
                self.alpha_low,
                self.alpha_high,
                self.gain_low,
                self.gain_mid,
                self.gain_high,
            )
            return

        _threeband_step_numpy(
            self._x,
            self._l0,
            self._l1,
            self._l2,
            self._l3,
            self._h0,
            self._h1,
            self._h2,
            self._h3,
            self.data_low,
            self.data_mid,
            self.data_high,
            self.data_equalized,
            self.alpha_low,
            self.alpha_high,
            self.gain_low,
            self.gain_mid,
            self.gain_high,
        )

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

            if not self.output_low.full():
                self.output_low.put_nowait((data_time, self.data_low))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "ThreeBand:output_low queue is full"))

            if not self.output_mid.full():
                self.output_mid.put_nowait((data_time, self.data_mid))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "ThreeBand:output_mid queue is full"))

            if not self.output_high.full():
                self.output_high.put_nowait((data_time, self.data_high))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "ThreeBand:output_high queue is full"))

            if not self.output.full():
                self.output.put_nowait((data_time, self.data_equalized))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "ThreeBand:output queue is full"))

            num_cubes += 1
            current_time = time.time()
            if (current_time - last_time) >= 5.0:
                self.measured_cps = num_cubes / 5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"ThreeBand:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"ThreeBand:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_time = current_time
