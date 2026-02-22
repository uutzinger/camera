## Installation

- [Installation](#installation)
  - [This Package](#this-package)
  - [OpenCV](#opencv)
  - [Picamera2](#picamera2)
  - [Spinnaker (PySpin)](#spinnaker-pyspin)
  - [Pylon (PyPylon)](#pylon-pypylon)
  - [TIFF and HDF5](#tiff-and-hdf5)

### This Package

If you cloned the repository:
- ``` cd "folder where you have this Readme.md file" ```
- ``` pip install -e . ``` or  ```python setup.py bdist_wheel``` and ```pip3 install .\dist\*.whl```

Directly from pypi:
- ```pip install camera-util```

### OpenCV

For many cameras.

*Unix*: 
system level installation ```sudo apt install libopencv-dev python3-opencv```  
python level isntallation ```pip3 install opencv-contrib-python``` 
Third party installation helper:
[Qengineering (Ubuntu/Jetson)](https://qengineering.eu/install-opencv-on-jetson-nano.html) 
[Qengineering (Raspberry Pi)](https://qengineering.eu/install-opencv-on-raspberry-pi.html) 
or for Ubuntu you can also use my install script: [configure_opencv.sh](https://github.com/uutzinger/Linux_InstallScripts/blob/main/configure_opencv.sh)

*Windows*: 
```pip3 install opencv-contrib-python```


### Picamera2

Raspberry Pi: use `piCamera2Capture` (Picamera2/libcamera).

```bash
sudo apt install -y python3-picamera2
```

### Spinnaker (PySpin)

For Teledyne/FLIR Spinnaker cameras (including Blackfly models), install [Spinnaker SDK and PySpin](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads)

Extract the files.
*Windows*: run the exe file. 
*Unix*: ```.\install_spinnaker.sh``` and ```pip install spinnaker_python...```

Please note, python bindings are are available for python version one or two generations behind the latest.

### Pylon (PyPylon)

For Basler cameras, install the Basler runtime first, then Python bindings.

The Runtime you need to download from [Basler website](https://www.baslerweb.com/en/downloads/software/?downloadCategory.values.label.data=pylon).

The python bindings you install with:

```bash
# Python bindings
pip install pypylon
```

Additional packages (only if needed):
- **blaze cameras**: install the **pylon Supplementary Package for blaze**.
- **pylon Data Processing / vTools**: install **CodeMeter Runtime**.
- **pylon Event Logger**: optional, useful for diagnostics only (not required for normal capture).

### TIFF and HDF5

For recording and storing image data in tiff files. The tiff format can record multiple time sequential images in a single tiff file.

On Unix ```sudo apt install libhdf5-dev libtiff-dev```

All systems: ```pip install tifffile imagecodecs h5py```

To get better tiff performance on Windows, [building libtiff is advised](https://github.com/uutzinger/Windows_Install_Scripts/blob/master/Buildinglibtiff.md)

### GStreamer <!-- omit from toc -->

Used for RTSP/RTP pipelines and the `gCapture` appsink backend.

*Linux / Ubuntu*:
```bash
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-rtsp gstreamer1.0-alsa gstreamer1.0-pulseaudio v4l-utils
```

Linux (Python bindings for direct GStreamer use, needed by `gCapture`):

```bash
sudo apt install python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0
```

Verify Python GI works:

```bash
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; Gst.init(None); print(Gst.version_string())"
```

*Windows (GStreamer runtime)*:

Install GStreamer from [GStreamer](https://gstreamer.freedesktop.org/data/pkg/windows/) (typically the MSVC x86_64 installer). Add the GStreamer `bin` folder (e.g. `C:\gstreamer\1.0\msvc_x86_64\bin` or `C:\gstreamer\1.0\x86_64\bin`, depending on installer) to your `PATH`.

Verify GStreamer works:

```powershell
gst-inspect-1.0 --version
gst-device-monitor-1.0 Video/Source
gst-inspect-1.0 ksvideosrc
gst-inspect-1.0 dshowvideosrc
```

`gCapture` (and the GI backend of `rtpCapture`/`rtspCapture`) depend on Python GI bindings (`import gi`) which is not directly supported by default on Windows Options:

- MSYS2 Python + PyGObject + GStreamer (then run your scripts from that MSYS2 environment)
- Conda / Miniconda with conda-forge packages (example):

```powershell
conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly
```

If you do not want MSYS2/Conda on Windows:

- Use `rtspCapture` via OpenCV (FFmpeg backend) for RTSP cameras.
- For raw RTP/UDP (`rtpCapture`), use the FFmpeg fallback and provide `rtp_sdp`.

### FFmpeg (RTP fallback) <!-- omit from toc -->

If you want `rtpCapture` to work on Windows/macOS without Python GI/GStreamer, install the bundled FFmpeg executable:

```bash
pip install imageio-ffmpeg
```

- The RTP fallback uses FFmpeg and requires an SDP file describing the RTP stream (set `rtp_sdp` in the capture config).
- The sender in `camera/streamer/rtpserver.py` can generate a matching SDP file when it uses the FFmpeg backend (pass `sdp_path=...`).
- RTSP does not need this; `rtspCapture` can usually use OpenCV+FFmpeg directly on an `rtsp://...` URL.
