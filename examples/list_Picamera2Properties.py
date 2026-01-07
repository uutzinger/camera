import sys
import pprint

try:
    from picamera2 import Picamera2
except Exception as exc:
    print("Picamera2 not available:", exc)
    sys.exit(1)

pp = pprint.PrettyPrinter(indent=2, width=100, compact=True)

picam2 = Picamera2()

print("\n=== Camera Properties ===")
props = getattr(picam2, "camera_properties", {})
pp.pprint(props)

print("\n=== Sensor Modes ===")
sensor_modes = getattr(picam2, "sensor_modes", [])
for idx, mode in enumerate(sensor_modes):
    print(f"[{idx}]", mode)

print("\n=== Control Ranges (min/max/step/default) ===")
controls = getattr(picam2, "camera_controls", {})
if isinstance(controls, dict):
    # Focus on common controls first
    focus = [
        "AeEnable", "ExposureTime", "FrameRate", "AeMeteringMode", "AeExposureMode",
        "AwbEnable", "ColourTemperature", "ColourGains",
        "AnalogueGain", "AfMode", "LensPosition",
        "Brightness", "Contrast", "Saturation", "Sharpness",
        "NoiseReductionMode", "ScalerCrop"
    ]
    def fmt_range(r):
        try:
            if isinstance(r, dict):
                return {
                    k: r.get(k) for k in ("min", "max", "step", "default")
                }
            return r
        except Exception:
            return r
    for name in focus:
        rng = controls.get(name)
        if rng is not None:
            print(f"- {name}:", fmt_range(rng))
    # Also list any other available controls
    others = [k for k in controls.keys() if k not in focus]
    if others:
        print("\nOther controls:")
        for k in sorted(others):
            print(f"- {k}:", fmt_range(controls.get(k)))
else:
    print("No camera_controls dictionary available")

print("\n=== Current Values (after temporary start) ===")
try:
    # Configure a safe mode quickly to read metadata values
    cfg = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"},
                                            controls={"FrameRate": 30})
    picam2.configure(cfg)
    picam2.start()
    meta = picam2.capture_metadata() or {}
    # Show a subset of live values
    for key in [
        "AeEnable", "ExposureTime", "FrameRate", "AeMeteringMode", "AeExposureMode",
        "AwbEnable", "ColourTemperature", "ColourGains",
        "AnalogueGain", "AfMode", "LensPosition"
    ]:
        if key in meta:
            print(f"- {key}: {meta.get(key)}")
except Exception as e:
    print("Could not read live metadata:", e)
finally:
    try:
        picam2.stop()
    except Exception:
        pass

print("\n=== Probe Candidate Size/FPS Pairs ===")
candidates = [((320,240),120), ((640,480),90), ((1280,720),60), ((1920,1080),30)]
for size, fps in candidates:
    try:
        cfg = picam2.create_video_configuration(main={"size": size, "format": "BGR888"},
                                                controls={"FrameRate": fps})
        picam2.configure(cfg)
        print("OK:", size, fps)
    except Exception as e:
        print("No:", size, fps, "-", e)

picam2.close()
