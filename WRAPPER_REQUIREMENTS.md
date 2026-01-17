# Functional Requirements - Camera Capture Wrappers

Scope: This document captures functional requirements for an example
camera wrapper based on the Picamera2 wrappers and defines cross-wrapper
requirements to guide refactoring of other wrapper such as
`cv2capture` and `blackflycapture`. Sources: `camera/capture/picamera2core.py`,
`camera/capture/picamera2capture.py`, `camera/capture/picamera2captureQt.py`.

## Cross-Wrapper Requirements (Common Interface and Behavior)

1) Core/Wrapper Split
   - Provide a non-threaded core that owns device configuration, format
     selection, and the actual frame acquisition.
   - Provide a wrapper that owns a capture loop thread (not a `Thread` subclass)
     and exposes a polling buffer API to consumers.
   - Provide a Qt wrapper variant when GUI integration is required.

2) Buffering Model
   - Use a SPSC ring buffer for frames, not a Queue.
   - Expose `buffer` as the primary FrameBuffer and `capture` as an alias.
   - Provide overwrite behavior when full (configurable), with best-effort
     logging when frames are dropped.

3) Capture Lifecycle
   - The wrapper must allow starting/stopping capture without tearing down the
     capture loop thread (loop continues to run).
   - It must also provide paus and resume.
   - A `close` path must stop the loop and release resources.
   - Capture state should be reflected in a core-level `capturing` flag.

4) Reconfigure-on-the-fly
   - Resolution changes and flip changes must be supported during capture.
   - If capture is running, reconfigure must pause capture, apply changes, update
     buffer allocation if needed, and resume automatically.
   - Reconfigure must be safe if invoked while the loop thread is active.

5) Controls and Config
   - Support a `configs` dict that defines capture size, fps, formats, buffering,
     and camera controls
   - Camera controls must include exposure, autoexposure, whitebalance and autowhite balance, etc.
   - Expose setters and properties for common controls in the wrapper such as 
     - width, height, size, resoltuion
     - exposure, gain, autoexposure, whitebalance autowitebalance
     - brightness, contrast satureation if appropriate
     - sharpness and noise reduction if appropriate
   - and a generic
     `set_controls` for advanced/driver-specific controls.
   - Provide a read-only `get_control` access path for control/metadata values.

6) Timestamps and Metadata
   - Frame delivery must include a timestamp in milliseconds (float) derived from
     device metadata when available, with a fallback to a monotonic clock.

7) Error Handling and Logging
   - Log capture errors and buffer drop warnings via a queue (non-Qt) or signal
     (Qt), using Python logging levels.
   - Failures to set controls or reconfigure should be best-effort and logged.

8) Format Conversion and Output Shape
   - The core must normalize frame output to an expected numpy shape/dtype after
     any device/SDK conversion, and the wrapper must allocate buffers to match.

9) API Parity
   - The non-Qt and Qt wrappers must expose a compatible API surface so the
     consumer can switch wrappers with minimal code changes.

## Picamera2

### Picamera2 Core Requirements (PiCamera2Core)

1) Core API
   - Provide `open_cam()`, `close_cam()`, `capture_array()`, `get_control()`,
     `set_controls()`, `log_stream_options()` and helpers for supported formats.

2) Stream Selection
   - Support main (processed) and raw (Bayer) streams, with a stream policy
     controlling how sensor modes are selected.

3) Format Mapping
   - Accept `format`, `fourcc`, `main_format`, and `raw_format` and map them to
     supported Picamera2/libcamera formats.

4) Output Scaling and Flip
   - Support hardware transform where possible; otherwise apply CPU resize/flip
     to achieve requested output behavior.

5) Buffer Allocation
   - Allocate a FrameBuffer sized to the configured output and format.
   - Support configurable capacity and overwrite behavior.

6) Synthetic/Test Mode
   - Provide a synthetic/test-pattern mode that emits frames without Picamera2,
     while maintaining consistent output shape and type.

7) Camera Controls
   - Apply configuration-time controls (exposure, AE/AWB, focus, tuning, fps) and
     allow runtime control updates via `set_controls`.

### Picamera2 Threaded Wrapper Requirements (piCamera2Capture)

1) Threaded Capture Loop
   - Own a loop thread that pulls frames from the core and pushes into the ring
     buffer without blocking.

2) Buffer Management
   - Expose `buffer` and `capture` and update them whenever the core is
     reconfigured or a new buffer is allocated.

3) Reconfigure Handling
   - Track pending flip/resolution changes and apply them inside the loop with
     safe pause/restart semantics.

4) Control Helpers
   - Provide setters for exposure, AE, fps, AE metering, AWB and AWB mode, flip,
     and resolution.

5) Performance Stats
   - Compute measured fps periodically and log it to the wrapper log queue.

### Picamera2 Qt Wrapper Requirements (piCamera2CaptureQt)

1) Qt Signals
   - Provide signals for stats, log, opened, started, and stopped.

2) Capture Loop
   - Use the same loop design as the non-Qt wrapper, but emit signals in place
     of log queue usage.

3) API Parity
   - Provide the same control setters and query helpers as the non-Qt wrapper.

4) QImage Conversion
   - Provide a `convertQimage(frame)` helper that accepts OpenCV BGR arrays and
     returns a compatible QImage.

## OpenCV

### CV2 Core Requirements (cv2Core)

1) Core API
   - Provide `open_cam()` and `close_cam()` for `cv2.VideoCapture` lifecycle.
   - Provide low-level `_set_prop()`/`_get_prop()` helpers guarded by a lock.

2) Platform Backend Selection
   - Select the optimal OpenCV backend by OS (Windows: DSHOW, macOS: AVFOUNDATION,
     Linux: V4L2, fallback: CAP_ANY).

3) Config Normalization
   - Initialize from `configs` and explicit overrides (`camera_num`, `res`,
     `exposure`) with sensible defaults when missing.
   - Recognize config keys: `camera_res`, `output_res`, `fps`, `flip`,
     `buffersize`, `fourcc`, `autoexposure`, `gain`, `wb_temp`, `autowb`,
     and `settings`.

4) Camera Properties
   - Expose getters/setters for width, height, resolution, exposure, autoexposure,
     fps, fourcc, buffersize, gain, white balance temperature, and auto white
     balance.
   - Use safe numeric conversions for unsupported OpenCV properties
     (normalize NaN/None to sentinel values).

5) Exposure Robustness
   - Provide a helper that applies autoexposure and exposure in a
     backend-tolerant way, with best-effort readback verification and logging.

6) Settings Dialog
   - Provide a method that invokes `CAP_PROP_SETTINGS` (best-effort, backend-
     dependent) for interactive camera settings.

## CV2 Threaded Wrapper Requirement (cv2Capture)

1) Threaded Capture Loop
   - Own a capture thread that reads frames via `VideoCapture.read()` and pushes
     `(frame, timestamp_ms)` into a FrameBuffer (SPSC ring buffer).

2) Buffering and Drop Behavior
   - Use a FrameBuffer with configurable capacity and overwrite behavior.
   - Expose `buffer` as the primary FrameBuffer and `capture` as an alias.
   - Log a warning when the buffer is full and frames are dropped.

3) Frame Processing
   - Apply optional output resize and flip/rotation transformations based on
     configuration.

4) Lifecycle
   - Provide `start()`/`stop()` to control the capture thread and a `close_cam()`
     path that releases the underlying `VideoCapture`.

5) Stats
   - Compute a measured FPS periodically and log it via the log queue.

## CV2 Qt Wrapper Requirements (cv2CaptureQt)

1) Qt Integration
   - Provide a Qt wrapper that mirrors the non-Qt API and emits Qt signals for
     log and stats (instead of a log queue).

2) Capture Loop
   - Use the same capture loop semantics as the non-Qt wrapper (FrameBuffer,
     timestamps, resize/flip).

3) API Parity
   - Expose the same control helpers and property accessors as the non-Qt wrapper
     to minimize consumer changes.

## BlackFly Core Requirements (blackflyCore)

## BlackFly Threaded Wrapper Requirement (blackflyCapture)

## BlackFly Qt Wrapper Requirements (blackflyCaptureQt)
