import sys
import pprint
import logging

try:
    from picamera2 import Picamera2
except Exception as exc:
    print("Picamera2 not available:", exc)
    sys.exit(1)

logging.basicConfig(level=logging.WARNING)  # options are: DEBUG, INFO, ERROR, WARNING

# Silence Picamera2 / libcamera logs; keep only this script's logging output
for _name in ("picamera2", "libcamera"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

pp = pprint.PrettyPrinter(indent=2, width=100, compact=True)

picam2 = Picamera2()

print("\n=== Camera Properties ===")
props = getattr(picam2, "camera_properties", {})
pp.pprint(props)

print("\n=== Sensor Modes ===")
sensor_modes = getattr(picam2, "sensor_modes", [])
# Determine sensor active size/aspect
props_active = getattr(picam2, "camera_properties", {}) or {}
act_w = act_h = None
paa = props_active.get("PixelArrayActiveAreas") or props_active.get("ActiveArea")
if isinstance(paa, (list, tuple)) and len(paa) > 0 and isinstance(paa[0], (list, tuple)) and len(paa[0]) == 4:
    act_w, act_h = int(paa[0][2]), int(paa[0][3])
else:
    pas = props_active.get("PixelArraySize")
    if isinstance(pas, (list, tuple)) and len(pas) == 2:
        act_w, act_h = int(pas[0]), int(pas[1])

def _ar(w,h):
    try:
        return round(float(w)/float(h), 4)
    except Exception:
        return None

sensor_ar = _ar(act_w, act_h) if (act_w and act_h) else None

for idx, mode in enumerate(sensor_modes):
    # mode is typically a dict, try to extract fields
    size = mode.get('size') if isinstance(mode, dict) else None
    bit_depth = mode.get('bit_depth') if isinstance(mode, dict) else None
    fps = mode.get('fps') if isinstance(mode, dict) else None
    fmt = mode.get('format') if isinstance(mode, dict) else None
    if isinstance(size, (list, tuple)) and len(size) == 2:
        mw, mh = int(size[0]), int(size[1])
        mar = _ar(mw, mh)
        fov_note = []
        if sensor_ar and mar and abs(mar - sensor_ar) > 0.02:
            fov_note.append('aspect-crop')
        if act_w and act_h and (mw < act_w and mh < act_h):
            fov_note.append('windowed')
        note = (f"; FOV: {', '.join(fov_note)}" if fov_note else "")
        print(f"[{idx}] size={mw}x{mh} ar={mar} fps={fps} bit_depth={bit_depth} format={fmt}{note}")
    else:
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

print("\n=== Preview Crop Probe (processed stream) ===")
def probe_preview(size):
    try:
        cfg = picam2.create_video_configuration(main={"size": size, "format": "BGR888"},
                                                controls={"FrameRate": 30})
        picam2.configure(cfg)
        # Try to force full-FOV for processed stream
        try:
            props = getattr(picam2, "camera_properties", {})
            crop_rect = None
            paa = props.get("PixelArrayActiveAreas") or props.get("ActiveArea")
            if isinstance(paa, (list, tuple)) and len(paa) > 0 and isinstance(paa[0], (list, tuple)) and len(paa[0]) == 4:
                crop_rect = (int(paa[0][0]), int(paa[0][1]), int(paa[0][2]), int(paa[0][3]))
            elif act_w and act_h:
                crop_rect = (0, 0, act_w, act_h)
            if crop_rect:
                picam2.set_controls({"ScalerCrop": crop_rect})
        except Exception:
            pass
        picam2.start()
        meta = picam2.capture_metadata() or {}
        sc = meta.get('ScalerCrop')
        print(f"preview {size}: ScalerCrop={sc} (full={(0,0,act_w,act_h) if (act_w and act_h) else None})")
    except Exception as e:
        print(f"preview {size}: failed - {e}")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass

for size in [(320,240),(640,480),(1280,720)]:
    probe_preview(size)

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
