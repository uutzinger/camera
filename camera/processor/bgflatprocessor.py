###############################################################################
# Background and flatfield processor
# 
# Urs Utzinger 2022,26
# GPT5.2 2026
#
# Changes:
# 2025 Code review and cleanup
# 2022 Initial release
###############################################################################

from __future__ import annotations

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


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _sample_frame_intensity_numba(
        cube: np.ndarray,
        dy: int,
        dx: int,
        out: np.ndarray,
    ) -> None:
        """Sample frame intensity by summing pixels at intervals of (dy, dx)."""
        depth, height, width = cube.shape
        for i in prange(depth):
            total = np.uint64(0)
            for y in range(0, height, dy):
                for x in range(0, width, dx):
                    total += np.uint64(cube[i, y, x])
            out[i] = total

    @njit(cache=True, fastmath=True, parallel=True)
    def _bgflat_apply_numba(
        cube: np.ndarray,
        background: np.ndarray,
        flatfield: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Apply background subtraction and flatfield correction to a data cube."""
        depth, height, width = cube.shape
        for i in prange(depth):
            for y in range(height):
                for x in range(width):
                    value = int(cube[i, y, x]) - int(background[y, x])
                    if value < 0:
                        value = 0
                    corrected = value * int(flatfield[y, x])
                    if corrected > 65535:
                        corrected = 65535
                    out[i, y, x] = np.uint16(corrected)

def _normalize_bg_delta(bg_delta: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize bg_delta to (dy, dx) tuple."""
    if isinstance(bg_delta, int):
        d = max(1, int(bg_delta))
        return d, d
    if len(bg_delta) != 2:
        raise ValueError("bg_delta must be int or tuple(dy, dx)")
    dy = max(1, int(bg_delta[0]))
    dx = max(1, int(bg_delta[1]))
    return dy, dx

def _prepare_flatfield_uint16(
    flatfield: np.ndarray | int | float,
    shape_hw: tuple[int, int],
    input_max: int,
) -> tuple[np.ndarray, float, bool]:
    """Prepare 2D flatfield as uint16 multipliers.

    Returns:
      ff_u16: uint16 flatfield multipliers to apply after background subtraction.
      scale : float multiplier applied to flatfield to preserve dynamic range without guaranteed overflow.
      may_clip : True if flatfield may cause clipping at uint16 output range.
    """
    height, width = shape_hw
    if input_max <= 0:
        input_max = 255

    if np.isscalar(flatfield):
        ff_in = np.full((height, width), flatfield, dtype=np.float64)
    else:
        ff_in = np.asarray(flatfield)
        if ff_in.ndim != 2 or ff_in.shape != (height, width):
            raise ValueError(
                f"flatfield must have shape ({height}, {width}), got {ff_in.shape}"
            )

    if np.issubdtype(ff_in.dtype, np.floating):
        ff_float = np.nan_to_num(ff_in.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(ff_float, 0.0, out=ff_float)
        max_ff = float(np.max(ff_float)) if ff_float.size else 0.0
        if max_ff <= 0.0:
            return np.zeros((height, width), dtype=np.uint16), 0.0, False

        # Choose integer scale that preserves dynamic range without guaranteed overflow.
        max_multiplier = 65535.0 / float(input_max)
        scale = float(np.floor(max_multiplier / max_ff))
        may_clip = scale < 1.0
        if scale < 1.0:
            scale = 1.0

        ff_scaled = np.rint(ff_float * scale)
        np.clip(ff_scaled, 0.0, 65535.0, out=ff_scaled)
        return ff_scaled.astype(np.uint16, copy=False), scale, may_clip

    ff_int = np.asarray(ff_in)
    if np.issubdtype(ff_int.dtype, np.signedinteger):
        ff_int = np.maximum(ff_int, 0)
    ff_u16 = np.clip(ff_int, 0, 65535).astype(np.uint16, copy=False)
    max_mul = int(np.max(ff_u16)) if ff_u16.size else 0
    may_clip = (max_mul * int(input_max)) > 65535
    return ff_u16, 1.0, may_clip

def _sample_frame_intensity_numpy(cube: np.ndarray, dy: int, dx: int, out: np.ndarray) -> None:
    out[:] = np.sum(cube[:, ::dy, ::dx], axis=(1, 2), dtype=np.uint64)

def _bgflat_apply_numpy(
    cube: np.ndarray,
    background: np.ndarray,
    flatfield: np.ndarray,
    out: np.ndarray,
    tmp_i32: np.ndarray,
) -> None:
    np.subtract(cube.astype(np.int32), background[np.newaxis, :, :].astype(np.int32), out=tmp_i32)
    np.maximum(tmp_i32, 0, out=tmp_i32)
    np.multiply(tmp_i32, flatfield[np.newaxis, :, :].astype(np.int32), out=tmp_i32)
    np.clip(tmp_i32, 0, 65535, out=tmp_i32)
    out[:] = tmp_i32.astype(np.uint16)

def sort_background_first(
    data_cube: np.ndarray,
    background_index: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Roll 3D cube so the selected background frame becomes frame index 0."""
    cube = np.asarray(data_cube)
    if cube.ndim != 3:
        raise ValueError("data_cube must be 3D (depth, height, width)")
    depth = int(cube.shape[0])
    if depth < 1:
        raise ValueError("data_cube must have depth >= 1")

    bg = int(background_index) % depth
    if out is None:
        out_arr = np.empty_like(cube)
    else:
        out_arr = out
        if out_arr.shape != cube.shape or out_arr.dtype != cube.dtype:
            raise ValueError("out must match data_cube shape and dtype")

    if bg == 0:
        np.copyto(out_arr, cube)
        return out_arr

    keep = depth - bg
    out_arr[:keep] = cube[bg:]
    out_arr[keep:] = cube[:bg]
    return out_arr

def bgflat(
    data_cube: np.ndarray,
    background: np.ndarray,
    flatfield: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility helper: apply background subtraction + flatfield."""
    cube = np.asarray(data_cube)
    bg = np.asarray(background)
    ff = np.asarray(flatfield, dtype=np.uint16)

    if cube.ndim != 3:
        raise ValueError("data_cube must be 3D (depth, height, width)")
    if bg.shape != cube.shape[1:]:
        raise ValueError(f"background must have shape {cube.shape[1:]}, got {bg.shape}")
    if ff.shape != cube.shape[1:]:
        raise ValueError(f"flatfield must have shape {cube.shape[1:]}, got {ff.shape}")

    if out is None:
        out_arr = np.empty(cube.shape, dtype=np.uint16)
    else:
        out_arr = out
        if out_arr.shape != cube.shape or out_arr.dtype != np.uint16:
            raise ValueError("out must be uint16 and match data_cube shape")

    if _NUMBA_AVAILABLE:
        _bgflat_apply_numba(cube, bg, ff, out_arr)
    else:
        tmp_i32 = np.empty(cube.shape, dtype=np.int32)
        _bgflat_apply_numpy(cube, bg, ff, out_arr, tmp_i32)
    return out_arr


class bgflatProcessor(Thread):
    """Background removal + flatfield correction.

    The background image is auto-detected as the frame in a cube with the
    smallest sampled intensity. Optionally, the cube is rolled so this
    background frame becomes frame index 0 before correction.

    For each frame:

      corrected = max(frame - background, 0) * flatfield_uint16

    output is clipped to uint16 range.
    """

    def __init__(
        self,
        flatfield: np.ndarray | int | float,
        res: tuple[int, int, int] | None = (14, 540, 720),
        bg_delta: int | tuple[int, int] = (64, 64),
        sort_background_first_frame: bool = True,
        queue_size: int = 32,
    ):
        self.logger = logging.getLogger("bgflatProcessor")

        if flatfield is None:
            raise ValueError("Need to provide flatfield")

        self._flatfield_src = flatfield
        self.ff_scale = 1.0
        self.ff = None

        self.bg_dy, self.bg_dx = _normalize_bg_delta(bg_delta)
        self.sort_background_first_frame = bool(sort_background_first_frame)

        self.inten = np.zeros(1, dtype=np.uint64)
        self.bg = np.zeros((1, 1), dtype=np.uint8)
        self.data_cube_corr = np.zeros((1, 1, 1), dtype=np.uint16)
        self._tmp_i32 = np.zeros((1, 1, 1), dtype=np.int32)
        self._cube_sorted = np.zeros((1, 1, 1), dtype=np.uint8)
        self.background_indx = 0
        self._cube_shape: tuple[int, int, int] | None = None
        self._input_max: int = 255
        self._cube_dtype = np.dtype(np.uint8)

        self.input = Queue(maxsize=queue_size)
        self.output = Queue(maxsize=queue_size)
        self.log = Queue(maxsize=64)

        if res is not None:
            if len(res) != 3:
                raise ValueError("res must be (depth, height, width)")
            depth, height, width = (int(v) for v in res)
            self._ensure_buffers((depth, height, width), input_max=255, cube_dtype=np.dtype(np.uint8))

        self.stopped = True
        self.measured_cps = 0.0
        self.measured_time = 0.0
        self._thread: Optional[Thread] = None
        self._input_queue = self.input
        self._output_queue = self.output

        Thread.__init__(self)

        impl = "numba" if _NUMBA_AVAILABLE else "numpy"
        if not self.log.full():
            self.log.put_nowait(
                (
                    logging.INFO,
                    (
                        f"BgFlat:impl={impl}, bg_delta=({self.bg_dy},{self.bg_dx}), "
                        f"sort_background_first={self.sort_background_first_frame}"
                    ),
                )
            )

    def _ensure_buffers(self, cube_shape: tuple[int, int, int], input_max: int, cube_dtype: np.dtype) -> None:
        """Allocate internal buffers if cube shape or input max value changed."""
        if (
            cube_shape == self._cube_shape
            and input_max == self._input_max
            and cube_dtype == self._cube_dtype
        ):
            return

        depth, height, width = cube_shape
        if depth < 1 or height < 1 or width < 1:
            raise ValueError(f"Invalid cube shape {cube_shape}")

        self.inten = np.zeros(depth, dtype=np.uint64)
        self.bg = np.zeros((height, width), dtype=np.uint8)
        self.data_cube_corr = np.zeros((depth, height, width), dtype=np.uint16)
        self._tmp_i32 = np.zeros((depth, height, width), dtype=np.int32)
        self._cube_sorted = np.zeros((depth, height, width), dtype=cube_dtype)

        ff_u16, scale, may_clip = _prepare_flatfield_uint16(
            flatfield=self._flatfield_src,
            shape_hw=(height, width),
            input_max=input_max,
        )
        self.ff = ff_u16
        self.ff_scale = float(scale)
        self._cube_shape = cube_shape
        self._input_max = int(input_max)
        self._cube_dtype = cube_dtype

        if hasattr(self, "log") and not self.log.full():
            self.log.put_nowait(
                (
                    logging.INFO,
                    f"BgFlat:flatfield_scale={self.ff_scale:.6g}, input_max={self._input_max}",
                )
            )
        if may_clip and hasattr(self, "log") and not self.log.full():
            self.log.put_nowait(
                (
                    logging.WARNING,
                    "BgFlat:flatfield may clip at uint16 output range",
                )
            )

    def stop(self):
        self.stopped = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def start(self, input_queue: Queue | None = None, output_queue: Queue | None = None):
        if not self.stopped:
            return

        self._input_queue = input_queue if input_queue is not None else self.input
        self._output_queue = output_queue if output_queue is not None else self.output

        self.stopped = False
        self._thread = Thread(target=self.update)
        self._thread.daemon = True
        self._thread.start()

    def _process_core(self, cube: np.ndarray) -> None:
        if _NUMBA_AVAILABLE:
            _sample_frame_intensity_numba(cube, self.bg_dy, self.bg_dx, self.inten)
        else:
            _sample_frame_intensity_numpy(cube, self.bg_dy, self.bg_dx, self.inten)

        self.background_indx = int(np.argmin(self.inten))
        if self.sort_background_first_frame:
            sort_background_first(cube, self.background_indx, out=self._cube_sorted)
            work_cube = self._cube_sorted
            np.copyto(self.bg, work_cube[0], casting="unsafe")
        else:
            work_cube = cube
            np.copyto(self.bg, work_cube[self.background_indx], casting="unsafe")

        if _NUMBA_AVAILABLE:
            _bgflat_apply_numba(work_cube, self.bg, self.ff, self.data_cube_corr)
        else:
            _bgflat_apply_numpy(work_cube, self.bg, self.ff, self.data_cube_corr, self._tmp_i32)

    def update(self):
        input_queue = self._input_queue
        output_queue = self._output_queue

        last_cps_time = time.time()
        num_cubes = 0
        total_time = 0.0

        while not self.stopped:
            current_time = time.time()
            if (current_time - last_cps_time) >= 5.0:
                self.measured_cps = num_cubes / 5.0
                self.measured_time = (total_time / num_cubes) if num_cubes else 0.0
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"BgFlat:CPS:{self.measured_cps}"))
                if not self.log.full():
                    self.log.put_nowait((logging.INFO, f"BgFlat:Time:{self.measured_time}"))
                num_cubes = 0
                total_time = 0.0
                last_cps_time = current_time

            try:
                cube_time, data_cube = input_queue.get(block=True, timeout=0.25)
            except Empty:
                continue

            cube = np.asarray(data_cube)
            if cube.ndim != 3:
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"BgFlat:expected 3D cube, got shape {cube.shape}"))
                continue
            if cube.dtype.kind not in ("u", "i"):
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, f"BgFlat:expected integer cube dtype, got {cube.dtype}"))
                continue
            if cube.dtype.kind == "i" and np.any(cube < 0):
                if not self.log.full():
                    self.log.put_nowait((logging.ERROR, "BgFlat:signed cube with negative values is not supported"))
                continue

            if not cube.flags.c_contiguous:
                cube = np.ascontiguousarray(cube)

            input_max = int(np.iinfo(cube.dtype).max) if cube.dtype.kind in ("u", "i") else 255
            self._ensure_buffers(cube.shape, input_max=input_max, cube_dtype=cube.dtype)

            start_time = time.perf_counter()
            self._process_core(cube)
            total_time += time.perf_counter() - start_time

            if not output_queue.full():
                output_queue.put_nowait((cube_time, self.data_cube_corr))
            elif not self.log.full():
                self.log.put_nowait((logging.WARNING, "BgFlat:Processed Output Queue is full!"))

            num_cubes += 1
