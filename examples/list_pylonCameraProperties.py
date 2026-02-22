##########################################################################
# List Basler pypylon camera properties and trigger capabilities
##########################################################################

import logging
import sys

try:
    from pypylon import pylon
except Exception as exc:
    print("pypylon not available:", exc)
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("PylonProps")


def _param(cam, name: str):
    return getattr(cam, name, None)


def _enum_choices(node) -> list[str]:
    if node is None:
        return []
    try:
        vals = list(node.Symbolics)
        return [str(v) for v in vals]
    except Exception:
        pass
    try:
        vals = list(node.GetSymbolics())
        return [str(v) for v in vals]
    except Exception:
        return []


def _value(node):
    if node is None:
        return None
    try:
        return node.GetValue()
    except Exception:
        pass
    try:
        return node.Value
    except Exception:
        pass
    try:
        return str(node.ToString())
    except Exception:
        return None


def _set_enum(node, value: str) -> bool:
    if node is None:
        return False
    try:
        node.SetValue(value)
        return True
    except Exception:
        pass
    try:
        node.Value = value
        return True
    except Exception:
        return False


def _print_param(cam, name: str, *, with_choices: bool = False):
    node = _param(cam, name)
    if node is None:
        return
    val = _value(node)
    if val is not None:
        logger.debug("%s: %s", name, val)
    if with_choices:
        choices = _enum_choices(node)
        if choices:
            logger.debug("%s choices: %s", name, choices)


def _print_trigger_capabilities(cam):
    trig_selector = _param(cam, "TriggerSelector")
    trigger_types = _enum_choices(trig_selector)
    if not trigger_types:
        logger.debug("TriggerSelector not available")
        return

    logger.debug("TriggerSelector choices: %s", trigger_types)
    for selector in trigger_types:
        if not _set_enum(trig_selector, selector):
            continue
        mode_choices = _enum_choices(_param(cam, "TriggerMode"))
        source_choices = _enum_choices(_param(cam, "TriggerSource"))
        activation_choices = _enum_choices(_param(cam, "TriggerActivation"))
        external_sources = [s for s in source_choices if s.lower().startswith("line") or s.lower().startswith("in")]
        logger.debug(
            "Trigger %s -> mode:%s source:%s activation:%s external:%s",
            selector,
            mode_choices,
            source_choices,
            activation_choices,
            external_sources,
        )


def main():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) == 0:
        print("No cameras detected")
        return

    for idx, dev in enumerate(devices):
        cam = pylon.InstantCamera(factory.CreateDevice(dev))
        try:
            cam.Open()
            logger.debug("Camera %d:", idx)
            logger.debug("=============================")
            logger.debug("Model: %s", dev.GetModelName())
            logger.debug("Serial: %s", dev.GetSerialNumber())
            logger.debug("Vendor: %s", dev.GetVendorName())
            logger.debug("DeviceClass: %s", dev.GetDeviceClass())

            # Stream / acquisition / format
            _print_param(cam, "AcquisitionMode", with_choices=True)
            _print_param(cam, "AcquisitionFrameRateEnable")
            _print_param(cam, "AcquisitionFrameRate")
            _print_param(cam, "ExposureMode", with_choices=True)
            _print_param(cam, "ExposureAuto", with_choices=True)
            _print_param(cam, "ExposureTime")
            _print_param(cam, "GainAuto", with_choices=True)
            _print_param(cam, "Gain")
            _print_param(cam, "BalanceWhiteAuto", with_choices=True)
            _print_param(cam, "GammaEnable")
            _print_param(cam, "Gamma")

            _print_param(cam, "Width")
            _print_param(cam, "Height")
            _print_param(cam, "OffsetX")
            _print_param(cam, "OffsetY")
            _print_param(cam, "BinningHorizontal")
            _print_param(cam, "BinningVertical")
            _print_param(cam, "PixelFormat", with_choices=True)
            _print_param(cam, "SensorBitDepth", with_choices=True)

            # Trigger / IO
            _print_param(cam, "LineSelector", with_choices=True)
            _print_param(cam, "LineMode", with_choices=True)
            _print_param(cam, "LineSource", with_choices=True)
            _print_param(cam, "LineInverter")
            _print_trigger_capabilities(cam)

        except Exception as exc:
            logger.error("Failed to read camera %d: %s", idx, exc)
        finally:
            try:
                cam.Close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
