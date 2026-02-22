## Processing Modules

Threaded processors live in `camera/processor/`.

### Processors

- `bgflatProcessor` (`camera/processor/bgflatprocessor.py`)
  - Detects background frame from sampled stack intensity.
  - Optional stack sort so background becomes first frame.
  - Applies background subtraction + flatfield correction.
  - Output dtype: `uint16`.

- `highpassProcessor` (`camera/processor/highpassprocessor.py`)
  - One-pole low/high-pass processor.
  - Separate queues for low-pass and high-pass outputs.

- `lowpassProcessor` (`camera/processor/highpassprocessor.py`)
  - One-pole low-pass-only processor.
  - Useful when only LP output is needed.

- `highpassProcessor` (`camera/processor/runningsumprocessor.py`)
  - Running-sum high-pass processor.
  - Also exposes low-pass queue/output.

- `lowpassProcessor` (`camera/processor/runningsumprocessor.py`)
  - Running-sum low-pass-only processor.
  - Matches running-sum backend behavior.

- `threebandProcessor` (`camera/processor/threebandprocessor.py`)
  - Cascaded one-pole (4-stage) low/mid/high splitter.
  - Optional equalized output (`gain_low`, `gain_mid`, `gain_high`).

### Utilities

- `utilities.py` (`camera/processor/utilities.py`)
  - Binning helpers for stacks with layout `(stack, height, width)`.
  - Square binning: `bin2`, `bin3`, `bin4`, `bin5`, `bin6`, `bin8`, `bin9`, `bin10`, `bin12`, `bin15`, `bin18`, `bin20`.
  - Generic binning:
    - `bin_square(arr, bin=...)`
    - `bin_rect(arr, bin_h=..., bin_w=...)`
    - `bin_by_size(arr, bin=...)`

### Benchmarks / Tests

Scripts are in `testing/`:

- `testing/benchmark_binning.py`
- `testing/benchmark_runningsumprocessor.py`
