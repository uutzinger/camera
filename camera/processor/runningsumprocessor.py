###############################################################################
# Running-sum processors
#
# Moving-average lowpass (window length D = delay):
#   lp[n] = (1/D) * sum_{k=0..D-1} x[n-k]
#
# Implemented in recursive running-sum form (mathematically equivalent FIR):
#   s[n]  = s[n-1] + x[n] - x[n-D]
#   lp[n] = s[n] / D
#   hp[n] = x[n] - lp[n]
#
# Frequency guidance for moving-average / running-sum filter:
# - First null of lp:      f0 ~= fs / D
# - Approx -3 dB of lp:    fc ~= 0.443 * fs / D
#   -> choose D ~= 0.443 * fs / fc
#
# Running Sum filter is related to poor mans lowpass filter and can be implented as FIR filter:
# - One-pole LP (y=(1-alpha)*y_prev + alpha*x):
#     b = np.array([alpha]), a = np.array([1.0, -(1.0-alpha)])
# - One-pole HP equivalent:
#     b = np.array([1.0-alpha, -(1.0-alpha)]), a = np.array([1.0, -(1.0-alpha)])
###############################################################################

from __future__ import annotations

import collections
import logging
import time
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

def running_sum_first_zero_hz(fs_hz: float, delay: int) -> float:
    """First spectral null of moving-average lowpass."""
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be > 0")
    return float(fs_hz) / float(delay)

def running_sum_delay_to_fc(fs_hz: float, delay: int) -> float:
    """Approximate -3 dB corner of moving-average lowpass."""
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be > 0")
    return 0.443 * float(fs_hz) / float(delay)

def running_sum_fc_to_delay(fs_hz: float, fc_hz: float) -> int:
    """Choose moving-average delay from target -3 dB corner."""
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be > 0")
    if fc_hz <= 0.0:
        raise ValueError("fc_hz must be > 0")
    return max(1, int(round((0.443 * float(fs_hz)) / float(fc_hz))))

if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _runsum_step_lp_hp_numba(
        x_flat: np.ndarray,
        x_delay_flat: np.ndarray,
        sum_flat: np.ndarray,
        lp_flat: np.ndarray,
        hp_flat: np.ndarray,
        inv_delay: np.float32,
    ) -> None:
        # Fused step:
        # s[n] = s[n-1] + x[n] - x[n-D], lp[n] = s[n]/D, hp[n] = x[n]-lp[n]
        n = x_flat.size
        for i in prange(n):
            s_i = (x_flat[i] - x_delay_flat[i]) + sum_flat[i]
            sum_flat[i] = s_i
            lp_i = s_i * inv_delay
            lp_flat[i] = lp_i
            hp_flat[i] = x_flat[i] - lp_i

    @njit(cache=True, fastmath=True, parallel=True)
    def _runsum_step_lp_numba(
        x_flat: np.ndarray,
        x_delay_flat: np.ndarray,
        sum_flat: np.ndarray,
        lp_flat: np.ndarray,
        inv_delay: np.float32,
    ) -> None:
        n = x_flat.size
        for i in prange(n):
            s_i = (x_flat[i] - x_delay_flat[i]) + sum_flat[i]
            sum_flat[i] = s_i
            lp_flat[i] = s_i * inv_delay


def _apply_rsum_lp_hp(
    x_n: np.ndarray,
    x_n_delay: np.ndarray,
    sum_prev: np.ndarray,
    lowpass_avg: np.ndarray,
    highpass: np.ndarray,
    inv_delay: np.float32,
) -> None:
    """Update running-sum, lowpass and highpass in place."""
    if _NUMBA_AVAILABLE:
        _runsum_step_lp_hp_numba(
            x_n.reshape(-1),
            x_n_delay.reshape(-1),
            sum_prev.reshape(-1),
            lowpass_avg.reshape(-1),
            highpass.reshape(-1),
            inv_delay,
        )
        return
    # Use highpass buffer as temporary to avoid additional allocations.
    np.subtract(x_n, x_n_delay, out=highpass)
    np.add(highpass, sum_prev, out=sum_prev)
    np.multiply(sum_prev, inv_delay, out=lowpass_avg)
    np.subtract(x_n, lowpass_avg, out=highpass)


def _apply_rsum_lp(
    x_n: np.ndarray,
    x_n_delay: np.ndarray,
    sum_prev: np.ndarray,
    lowpass_avg: np.ndarray,
    inv_delay: np.float32,
) -> None:
    """Update running-sum and lowpass in place."""
    if _NUMBA_AVAILABLE:
        _runsum_step_lp_numba(
            x_n.reshape(-1),
            x_n_delay.reshape(-1),
            sum_prev.reshape(-1),
            lowpass_avg.reshape(-1),
            inv_delay,
        )
        return
    # Use lowpass buffer as temporary before writing final lowpass.
    np.subtract(x_n, x_n_delay, out=lowpass_avg)
    np.add(lowpass_avg, sum_prev, out=sum_prev)
    np.multiply(sum_prev, inv_delay, out=lowpass_avg)


def _apply_rsum(x_n: np.ndarray, x_n_delay: np.ndarray, sum_prev: np.ndarray, out: np.ndarray) -> None:
    """Backward-compatible helper for running-sum update only."""
    if _NUMBA_AVAILABLE:
        _runsum_step_lp_numba(
            x_n.reshape(-1),
            x_n_delay.reshape(-1),
            sum_prev.reshape(-1),
            out.reshape(-1),
            np.float32(1.0),
        )
        return
    np.subtract(x_n, x_n_delay, out=out)
    np.add(out, sum_prev, out=out)


def _apply_highpass_from_avg(x_n: np.ndarray, lowpass_avg: np.ndarray, out: np.ndarray) -> None:
    """Backward-compatible helper for highpass-from-lowpass step."""
    np.subtract(x_n, lowpass_avg, out=out)


class highpassProcessor(Thread):
    """Running-sum highpass processor.

    - Input queue: `input` with `(ts, data)`
    - Output queues:
        - `output_highpass` with `(ts, hp)`
        - `output_lowpass` with `(ts, lp)`
    - `output` is an alias to `output_highpass` for backward compatibility.
    """

    def __init__(
        self,
        res: tuple[int, ...],
        delay: int = 1,
        queue_size: int = 32,
        backend: str | None = None,
    ):
        self.stopped = True

        if backend not in (None, "", "numba"):
            raise ValueError("Only running-sum backend is supported; scipy backend was removed")

        if delay < 1:
            raise ValueError("delay must be >= 1")
        self.delay = int(delay)
        self.inv_delay = np.float32(1.0 / float(self.delay))
        self._res = tuple(int(v) for v in res)

        self.data_running_sum = np.zeros(self._res, dtype=np.float32)
        self.data_lowpass = np.zeros(self._res, dtype=np.float32)
        self.data_highpass = np.zeros(self._res, dtype=np.float32)
        self._x = np.zeros(self._res, dtype=np.float32)

        self._delay_line = collections.deque(maxlen=self.delay)
        for _ in range(self.delay):
            self._delay_line.append(np.zeros(self._res, dtype=np.float32))

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
            self.log.put_nowait((logging.INFO, f"RunSumHP:impl={impl}, delay={self.delay}"))

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
        x_n_delay = self._delay_line.popleft()
        _apply_rsum_lp_hp(
            self._x,
            x_n_delay,
            self.data_running_sum,
            self.data_lowpass,
            self.data_highpass,
            self.inv_delay,
        )
        np.copyto(x_n_delay, self._x)
        self._delay_line.append(x_n_delay)

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

            if not self.output_highpass.full():
                self.output_highpass.put_nowait((data_time, self.data_highpass))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "RunSumHP:output_highpass queue is full"))

            if not self.output_lowpass.full():
                self.output_lowpass.put_nowait((data_time, self.data_lowpass))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "RunSumHP:output_lowpass queue is full"))

            num_cubes += 1
            current_time = time.time()
            if (current_time - last_time) >= 5.0:
                self.measured_cps = num_cubes / 5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"RunSumHP:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"RunSumHP:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_time = current_time


class lowpassProcessor(Thread):
    """Running-sum lowpass-only processor."""

    def __init__(
        self,
        res: tuple[int, ...],
        delay: int = 1,
        queue_size: int = 32,
        backend: str | None = None,
    ):
        self.stopped = True

        if backend not in (None, "", "numba"):
            raise ValueError("Only running-sum backend is supported; scipy backend was removed")

        if delay < 1:
            raise ValueError("delay must be >= 1")
        self.delay = int(delay)
        self.inv_delay = np.float32(1.0 / float(self.delay))
        self._res = tuple(int(v) for v in res)

        self.data_running_sum = np.zeros(self._res, dtype=np.float32)
        self.data_lowpass = np.zeros(self._res, dtype=np.float32)
        self._x = np.zeros(self._res, dtype=np.float32)

        self._delay_line = collections.deque(maxlen=self.delay)
        for _ in range(self.delay):
            self._delay_line.append(np.zeros(self._res, dtype=np.float32))

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
            self.log.put_nowait((logging.INFO, f"RunSumLP:impl={impl}, delay={self.delay}"))

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
        x_n_delay = self._delay_line.popleft()
        _apply_rsum_lp(
            self._x,
            x_n_delay,
            self.data_running_sum,
            self.data_lowpass,
            self.inv_delay,
        )
        np.copyto(x_n_delay, self._x)
        self._delay_line.append(x_n_delay)

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
                self.output.put_nowait((data_time, self.data_lowpass))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "RunSumLP:output queue is full"))

            num_cubes += 1
            current_time = time.time()
            if (current_time - last_time) >= 5.0:
                self.measured_cps = num_cubes / 5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"RunSumLP:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"RunSumLP:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_time = current_time


# Backward compatibility
runningsumProcessor = highpassProcessor
