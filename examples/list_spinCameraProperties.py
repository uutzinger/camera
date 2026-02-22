##########################################################################
# List Spinnaker (PySpin) camera properties
##########################################################################
import sys
import logging

try:
    import PySpin
except Exception as exc:
    print("PySpin not available:", exc)
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SpinnakerProps")


def _safe_enum(node):
    try:
        return node.GetCurrentEntry().GetSymbolic()
    except Exception:
        try:
            return node.ToString()
        except Exception:
            return None


def _safe_value(node):
    try:
        return node.GetValue()
    except Exception:
        try:
            return node.ToString()
        except Exception:
            return None


def _print_node(name, node):
    if node is None:
        return
    try:
        if PySpin.IsReadable(node):
            val = _safe_value(node)
            logger.debug("%s: %s", name, val)
    except Exception:
        pass


def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num = cam_list.GetSize()
    if num == 0:
        print("No cameras detected")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    for idx in range(num):
        cam = cam_list.GetByIndex(idx)
        try:
            cam.Init()
            logger.debug("Camera %d:", idx)
            logger.debug("=============================")

            # Basic device info
            nodemap = cam.GetTLDeviceNodeMap()
            for name in ("DeviceModelName", "DeviceSerialNumber", "DeviceVendorName", "DeviceVersion"):
                node = PySpin.CStringPtr(nodemap.GetNode(name))
                if node and PySpin.IsReadable(node):
                    logger.debug("%s: %s", name, node.GetValue())

            # Stream / acquisition / format
            _print_node("AcquisitionMode", getattr(cam, "AcquisitionMode", None))
            _print_node("AcquisitionFrameRateEnable", getattr(cam, "AcquisitionFrameRateEnable", None))
            _print_node("AcquisitionFrameRate", getattr(cam, "AcquisitionFrameRate", None))
            _print_node("ExposureMode", getattr(cam, "ExposureMode", None))
            _print_node("ExposureAuto", getattr(cam, "ExposureAuto", None))
            _print_node("ExposureTime", getattr(cam, "ExposureTime", None))
            _print_node("GainAuto", getattr(cam, "GainAuto", None))
            _print_node("Gain", getattr(cam, "Gain", None))
            _print_node("BalanceWhiteAuto", getattr(cam, "BalanceWhiteAuto", None))
            _print_node("GammaEnable", getattr(cam, "GammaEnable", None))
            _print_node("Gamma", getattr(cam, "Gamma", None))
            _print_node("IspEnable", getattr(cam, "IspEnable", None))

            _print_node("Width", getattr(cam, "Width", None))
            _print_node("Height", getattr(cam, "Height", None))
            _print_node("OffsetX", getattr(cam, "OffsetX", None))
            _print_node("OffsetY", getattr(cam, "OffsetY", None))
            _print_node("BinningHorizontal", getattr(cam, "BinningHorizontal", None))
            _print_node("BinningVertical", getattr(cam, "BinningVertical", None))
            _print_node("AdcBitDepth", getattr(cam, "AdcBitDepth", None))
            _print_node("PixelFormat", getattr(cam, "PixelFormat", None))

            # Trigger / IO
            _print_node("TriggerMode", getattr(cam, "TriggerMode", None))
            _print_node("TriggerSelector", getattr(cam, "TriggerSelector", None))
            _print_node("TriggerSource", getattr(cam, "TriggerSource", None))
            _print_node("TriggerActivation", getattr(cam, "TriggerActivation", None))
            _print_node("TriggerOverlap", getattr(cam, "TriggerOverlap", None))
            _print_node("TriggerDelay", getattr(cam, "TriggerDelay", None))
            _print_node("LineSelector", getattr(cam, "LineSelector", None))
            _print_node("LineMode", getattr(cam, "LineMode", None))
            _print_node("LineSource", getattr(cam, "LineSource", None))
            _print_node("LineInverter", getattr(cam, "LineInverter", None))

        except Exception as exc:
            logger.error("Failed to read camera %d: %s", idx, exc)
        finally:
            try:
                cam.DeInit()
            except Exception:
                pass

    cam_list.Clear()
    system.ReleaseInstance()


if __name__ == "__main__":
    main()
