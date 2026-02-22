###############################################################################
# Binning helper
#
# This module implements rectangular binning for 3D frame stacks (stack, height, width).
#
# Urs Utzinger 2022-26
# Gpt 5.2 2026
###############################################################################
#
# Expected input layout is:
#   (stack, height, width)
#
# Naming:
#   - `bin` means square bin size (`bin x bin`).
#   - `bin_h` means vertical bin size (height / y direction).
#   - `bin_w` means horizontal bin size (width / x direction).
# 
# Output layout is:
#   (stack, height // bin_h, width // bin_w)
#
# For non-divisible shapes:
#   - strict=False (default): crop border pixels.
#   - strict=True: raise ValueError.
###############################################################################

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False

# Keep historical dtype behavior for these square bins.
_HISTORICAL_UINT32_BINS = {12, 15, 18, 20}

if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _bin_rect_u16_kernel(arr: np.ndarray, bin_h: int, bin_w: int, out: np.ndarray) -> None:
        stack, out_h, out_w = out.shape
        for s in prange(stack):
            for oy in range(out_h):
                iy0 = oy * bin_h
                for ox in range(out_w):
                    ix0 = ox * bin_w
                    acc = 0
                    for dy in range(bin_h):
                        for dx in range(bin_w):
                            acc += int(arr[s, iy0 + dy, ix0 + dx])
                    if acc < 0:
                        acc = 0
                    if acc > 65535:
                        acc = 65535
                    out[s, oy, ox] = acc

    @njit(cache=True, fastmath=True, parallel=True)
    def _bin_rect_u32_kernel(arr: np.ndarray, bin_h: int, bin_w: int, out: np.ndarray) -> None:
        stack, out_h, out_w = out.shape
        for s in prange(stack):
            for oy in range(out_h):
                iy0 = oy * bin_h
                for ox in range(out_w):
                    ix0 = ox * bin_w
                    acc = 0
                    for dy in range(bin_h):
                        for dx in range(bin_w):
                            acc += int(arr[s, iy0 + dy, ix0 + dx])
                    if acc < 0:
                        acc = 0
                    out[s, oy, ox] = acc

def _validate_input(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D array (stack,height,width), got shape={a.shape}")
    if a.dtype.kind not in ("u", "i"):
        raise ValueError(f"Expected integer dtype input, got {a.dtype}")
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return a

def _normalize_bins(
    *,
    bin: int | None = None,
    bin_h: int | None = None,
    bin_w: int | None = None,
    bin_y: int | None = None,
    bin_x: int | None = None,
    factor: int | None = None,
) -> tuple[int, int]:
    """Resolve bin arguments.

    Backward-compatible aliases:
      - factor -> bin
      - bin_y -> bin_h
      - bin_x -> bin_w
    """
    if factor is not None:
        if bin is not None:
            raise ValueError("Use either `bin` or `factor`, not both")
        bin = int(factor)

    if bin_y is not None:
        if bin_h is not None:
            raise ValueError("Use either `bin_h` or `bin_y`, not both")
        bin_h = int(bin_y)

    if bin_x is not None:
        if bin_w is not None:
            raise ValueError("Use either `bin_w` or `bin_x`, not both")
        bin_w = int(bin_x)

    if bin is not None:
        if bin_h is not None or bin_w is not None:
            raise ValueError("Use either square `bin` or (`bin_h`, `bin_w`), not both")
        by = int(bin)
        bx = int(bin)
    else:
        by = 1 if bin_h is None else int(bin_h)
        bx = 1 if bin_w is None else int(bin_w)

    if by < 1 or bx < 1:
        raise ValueError("bin_h and bin_w must be >= 1")
    return by, bx

def _crop_for_bins(arr: np.ndarray, bin_h: int, bin_w: int, strict: bool) -> np.ndarray:
    stack, h, w = arr.shape
    if strict and ((h % bin_h) != 0 or (w % bin_w) != 0):
        raise ValueError(f"Input shape {arr.shape} not divisible by (bin_h, bin_w)=({bin_h}, {bin_w})")
    h2 = (h // bin_h) * bin_h
    w2 = (w // bin_w) * bin_w
    return arr[:, :h2, :w2]

def _default_out_dtype(arr: np.ndarray, bin_h: int, bin_w: int) -> np.dtype:
    max_in = int(np.iinfo(arr.dtype).max) if arr.dtype.kind in ("u", "i") else 255
    max_sum = max(0, max_in) * int(bin_h) * int(bin_w)
    return np.uint16 if max_sum <= 65535 else np.uint32

def _bin_rect_numpy(arr: np.ndarray, bin_h: int, bin_w: int, out_dtype: np.dtype) -> np.ndarray:
    stack, h, w = arr.shape
    out_h = h // bin_h
    out_w = w // bin_w
    acc = arr.astype(np.int64, copy=False)
    out = acc.reshape(stack, out_h, bin_h, out_w, bin_w).sum(axis=(2, 4), dtype=np.int64)
    if out_dtype == np.uint16:
        np.clip(out, 0, 65535, out=out)
        return out.astype(np.uint16, copy=False)
    np.clip(out, 0, np.iinfo(np.uint32).max, out=out)
    return out.astype(np.uint32, copy=False)

def bin_rect(
    arr: np.ndarray,
    *,
    bin: int | None = None,
    bin_h: int | None = None,
    bin_w: int | None = None,
    out_dtype: np.dtype | None = None,
    strict: bool = False,
    out: np.ndarray | None = None,
    # Backward-compatible aliases:
    factor: int | None = None,
    bin_y: int | None = None,
    bin_x: int | None = None,
) -> np.ndarray:
    """Rectangular binning for (stack,height,width) arrays."""
    a = _validate_input(arr)
    by, bx = _normalize_bins(
        bin=bin,
        bin_h=bin_h,
        bin_w=bin_w,
        bin_y=bin_y,
        bin_x=bin_x,
        factor=factor,
    )
    a = _crop_for_bins(a, bin_h=by, bin_w=bx, strict=strict)

    stack, h, w = a.shape
    out_h = h // by
    out_w = w // bx

    if out_dtype is None:
        out_dtype = _default_out_dtype(a, by, bx)
    out_dtype = np.dtype(out_dtype)
    if out_dtype not in (np.dtype(np.uint16), np.dtype(np.uint32)):
        raise ValueError(f"Unsupported out_dtype {out_dtype}; use uint16 or uint32")

    if out is None:
        out_arr = np.empty((stack, out_h, out_w), dtype=out_dtype)
    else:
        out_arr = out
        if out_arr.shape != (stack, out_h, out_w):
            raise ValueError(f"out shape mismatch, expected {(stack, out_h, out_w)}, got {out_arr.shape}")
        if out_arr.dtype != out_dtype:
            raise ValueError(f"out dtype mismatch, expected {out_dtype}, got {out_arr.dtype}")

    if _NUMBA_AVAILABLE:
        if out_dtype == np.uint16:
            _bin_rect_u16_kernel(a, int(by), int(bx), out_arr)
        else:
            _bin_rect_u32_kernel(a, int(by), int(bx), out_arr)
        return out_arr

    out_np = _bin_rect_numpy(a, by, bx, out_dtype)
    if out is not None:
        np.copyto(out_arr, out_np, casting="unsafe")
        return out_arr
    return out_np

def bin_square(
    arr: np.ndarray,
    bin: int,
    *,
    out_dtype: np.dtype | None = None,
    strict: bool = False,
    out: np.ndarray | None = None,
    factor: int | None = None,
) -> np.ndarray:
    """Square binning wrapper (bin_h = bin_w = bin)."""
    return bin_rect(
        arr,
        bin=bin,
        out_dtype=out_dtype,
        strict=strict,
        out=out,
        factor=factor,
    )

def bin2(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 2, out_dtype=np.uint16, strict=strict, out=out)

def bin3(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 3, out_dtype=np.uint16, strict=strict, out=out)

def bin4(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 4, out_dtype=np.uint16, strict=strict, out=out)

def bin5(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 5, out_dtype=np.uint16, strict=strict, out=out)

def bin6(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 6, out_dtype=np.uint16, strict=strict, out=out)

def bin7(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 7, out_dtype=np.uint16, strict=strict, out=out)

def bin8(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 8, out_dtype=np.uint16, strict=strict, out=out)

def bin9(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 9, out_dtype=np.uint16, strict=strict, out=out)

def bin10(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 10, out_dtype=np.uint16, strict=strict, out=out)

def bin11(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 11, out_dtype=np.uint16, strict=strict, out=out)

def bin12(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    # Keep historical dtype contract.
    return bin_square(arr, 12, out_dtype=np.uint32, strict=strict, out=out)

def bin13(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 13, out_dtype=np.uint32, strict=strict, out=out)

def bin14(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 14, out_dtype=np.uint32, strict=strict, out=out)

def bin15(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    # Keep historical dtype contract.
    return bin_square(arr, 15, out_dtype=np.uint32, strict=strict, out=out)

def bin16(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 16, out_dtype=np.uint32, strict=strict, out=out)

def bin17(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 17, out_dtype=np.uint32, strict=strict, out=out)

def bin18(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 18, out_dtype=np.uint32, strict=strict, out=out)

def bin19(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 19, out_dtype=np.uint32, strict=strict, out=out)

def bin20(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 20, out_dtype=np.uint32, strict=strict, out=out)

def bin21(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 21, out_dtype=np.uint32, strict=strict, out=out)

def bin22(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 22, out_dtype=np.uint32, strict=strict, out=out)

def bin23(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 23, out_dtype=np.uint32, strict=strict, out=out)

def bin24(arr: np.ndarray, *, strict: bool = False, out: np.ndarray | None = None) -> np.ndarray:
    return bin_square(arr, 24, out_dtype=np.uint32, strict=strict, out=out)

BIN_FUNCS = {
    2: bin2,
    3: bin3,
    4: bin4,
    5: bin5,
    6: bin6,
    7: bin7,
    8: bin8,
    9: bin9,
    10: bin10,
    12: bin12,
    15: bin15,
    18: bin18,
    20: bin20,
    21: bin21,
    22: bin22,
    23: bin23,
    24: bin24,
}

def bin_by_size(
    arr: np.ndarray,
    bin: int,
    *,
    strict: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Square-bin dispatcher by bin size."""
    fn = BIN_FUNCS.get(int(bin))
    if fn is None:
        return bin_square(arr, int(bin), strict=strict, out=out)
    return fn(arr, strict=strict, out=out)

def bin_by_factor(
    arr: np.ndarray,
    factor: int,
    *,
    strict: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible alias for `bin_by_size`."""
    return bin_by_size(arr, int(factor), strict=strict, out=out)

