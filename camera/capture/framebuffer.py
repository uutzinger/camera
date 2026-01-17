###############################################################################
# Frame Buffer
#
# Urs Utzinger
# GPT-5.2
# Claude-
###############################################################################
from __future__ import annotations

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
else:
    import numpy as np

from typing import Optional, Tuple

class FrameBuffer:
    """
    Single-producer / single-consumer (SPSC) ring buffer for fixed-shape NumPy frames.

    Notes:
      - Uses power-of-two capacity for fast masking.
      - Stores frames in a preallocated ndarray: (capacity, *frame_shape).
      - Can operate with overwrite=False (default): push fails if full.
        If overwrite=True: oldest frames are dropped when full (ring overwrite).

    init parameters:
        - capacity: number of frames the ring can hold (rounded up to power of two)
        - frame_shape: shape of one frame, e.g. (H, W) or (H, W, C)
        - dtype: numpy dtype for storage
        - overwrite: if True, newest data overwrites oldest when full
    
    methods:
        - push(frame: np.ndarray, ts_ms: float) -> bool
        - pull(copy: bool = True) -> tuple[Optional[np.ndarray], Optional[float]]
        - pull_batch(max_items: int, copy: bool = True, out: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]
        - clear() -> None
        - avail -> int
        - free -> int
        - capacity -> int
        - frame_shape -> Tuple[int, ...]
        - dtype -> np.dtype
    """

    __slots__ = (
        "_buf",
        "_buf_ts_ms",
        "_cap",
        "_mask",
        "_shape",
        "_dtype",
        "_head",
        "_tail",
        "_overwrite",
        "_scratch",
        "_scratch_ts",
    )

    def __init__(
        self,
        capacity: int,
        frame_shape: 'Tuple[int, ...]',
        dtype: 'np.dtype | str' = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        capacity: number of frames the ring can hold (rounded up to power of two)
        frame_shape: shape of one frame, e.g. (H, W) or (H, W, C)
        dtype: numpy dtype for storage (default: np.uint8, always 3-channel 8-bit for RGB in this implementation)
        overwrite: if True, newest data overwrites oldest when full
        Note: Only 3-channel 8-bit buffers are currently supported. For YUV/Bayer, adjust allocation accordingly.
        """
        if dtype is None:
            dtype = np.uint8
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not frame_shape:
            raise ValueError("frame_shape must be a non-empty tuple")

        cap = 1 << (int(capacity - 1).bit_length())  # next power of two
        self._cap = cap
        self._mask = cap - 1
        self._shape = tuple(int(x) for x in frame_shape)
        self._dtype = np.dtype(dtype)
        self._overwrite = bool(overwrite)

        # Preallocate buffer: (cap, *frame_shape)
        self._buf = np.empty((cap, *self._shape), dtype=self._dtype, order="C")
        self._buf_ts_ms = np.empty((cap,), dtype=np.float64, order="C")

        # Monotonic counters (unbounded ints). Only producer writes _head; only consumer writes _tail.
        self._head = 0
        self._tail = 0

        # Scratch buffer for pull_batch to return a contiguous block without realloc each time (optional use)
        self._scratch: 'Optional[np.ndarray]' = None
        self._scratch_ts: 'Optional[np.ndarray]' = None

    # ----------------------------
    # Introspection / sizing
    # ----------------------------

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def avail(self) -> int:
        """How many frames are currently available to read."""
        head = self._head
        tail = self._tail
        cap = self._cap
        if self._overwrite:
            min_tail = head - cap
            if tail < min_tail:
                tail = min_tail
        n = head - tail
        if n <= 0:
            return 0
        return cap if n >= cap else n

    @property
    def free(self) -> int:
        """How many free slots remain (0..capacity)."""
        free = self._cap - self.avail
        return 0 if free < 0 else free

    # ----------------------------
    # Producer side
    # ----------------------------
    def push(self, frame: np.ndarray, ts_ms: float | int | np.floating) -> bool:
        """
        Push one frame.
        Returns True if stored, False if dropped (when full and overwrite=False).
        """
        # Fast locals (reduce attribute lookups in the hot path)
        head = self._head
        tail = self._tail
        cap = self._cap
        if (head - tail) >= cap and (not self._overwrite):
            return False

        idx = head & self._mask
        buf = self._buf
        buf_ts = self._buf_ts_ms

        # Copy data into slot (avoids allocating new arrays).
        # np.copyto is usually fast and supports dtype conversion if needed.
        np.copyto(buf[idx], frame, casting="unsafe")
        try:
            # Numpy will cast to float64; avoid an extra Python float() call.
            buf_ts[idx] = ts_ms
        except Exception:
            buf_ts[idx] = 0.0

        # Publish: increment head ONLY after copy completes
        self._head = head + 1
        return True

    # ----------------------------
    # Consumer side
    # ----------------------------
    def pull(self, *, copy: bool = True) -> 'tuple[Optional[np.ndarray], Optional[float]]':
        """
        Pull one frame.
        If copy=True (default), returns a copy safe from being overwritten later.
        If copy=False, returns a view into the ring slot (FAST, but unsafe if producer overwrites).
        """
        tail = self._tail
        head = self._head
        cap = self._cap

        # If producer has overrun the consumer (overwrite mode), skip to the
        # oldest still-valid element.
        if self._overwrite:
            min_tail = head - cap
            if tail < min_tail:
                tail = min_tail
                self._tail = tail

        if tail >= head:
            return None, None

        idx = tail & self._mask
        frame_view = self._buf[idx]
        ts = float(self._buf_ts_ms[idx])
        self._tail = tail + 1

        if copy:
            return frame_view.copy(), ts
        else:
            return frame_view, ts

    def pull_batch(
        self,
        max_items: int,
        *,
        copy: bool = True,
        out: 'Optional[np.ndarray]' = None,
    ) -> 'tuple[np.ndarray, np.ndarray]':
        """
        Pull up to max_items frames.
        Returns an ndarray of shape (n, *frame_shape). n may be 0.

        copy=True: returns a contiguous copy (safe).
        copy=False: returns a stacked copy anyway (because a true view would be non-contiguous when wrapped).
                   If you want zero-copy, pull(copy=False) in a loop and process immediately.

        out: optional preallocated output array with shape (max_items, *frame_shape) and matching dtype.
             If provided, returns out[:n].
        """
        if max_items <= 0:
            return (
                np.empty((0, *self._shape), dtype=self._dtype),
                np.empty((0,), dtype=np.float64),
            )

        tail = self._tail
        head = self._head

        # If producer has overrun the consumer (overwrite mode), skip to the
        # oldest still-valid element.
        if self._overwrite:
            min_tail = head - self._cap
            if tail < min_tail:
                tail = min_tail
                self._tail = tail
        available = head - tail
        if available <= 0:
            return (
                np.empty((0, *self._shape), dtype=self._dtype),
                np.empty((0,), dtype=np.float64),
            )

        n = available if available < max_items else max_items
        cap = self._cap
        mask = self._mask
        buf = self._buf
        buf_ts_ms = self._buf_ts_ms

        if out is not None:
            if out.shape[:1] != (max_items,) or out.shape[1:] != self._shape:
                raise ValueError(f"out must have shape ({max_items}, {self._shape}), got {out.shape}")
            if out.dtype != self._dtype:
                raise ValueError(f"out dtype must be {self._dtype}, got {out.dtype}")
            dst = out
        else:
            # Reuse scratch if possible to avoid repeated allocations for same max_items
            if self._scratch is None or self._scratch.shape[0] < n or self._scratch.dtype != self._dtype:
                self._scratch = np.empty((max_items, *self._shape), dtype=self._dtype)
            dst = self._scratch

        if self._scratch_ts is None or self._scratch_ts.shape[0] < max_items:
            self._scratch_ts = np.empty((max_items,), dtype=np.float64)
        dst_ts = self._scratch_ts

        # Copy in at most two chunks (no wrap + wrap)
        start = tail & mask
        first = min(n, cap - start)
        second = n - first

        # Copy chunk(s)
        dst0 = dst[:first]
        np.copyto(dst0, buf[start:start + first])
        dst_ts[:first] = buf_ts_ms[start:start + first]

        if second:
            dst1 = dst[first:first + second]
            np.copyto(dst1, buf[0:second])
            dst_ts[first:first + second] = buf_ts_ms[0:second]

        # Advance tail after copies complete
        self._tail = tail + n

        # Return exact-size view
        batch = dst[:n]
        ts_batch = dst_ts[:n]
        return batch, ts_batch

    def clear(self) -> None:
        """Drop all pending frames."""
        self._tail = self._head

__all__ = ["FrameBuffer"]
