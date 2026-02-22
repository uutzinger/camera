//=============================================================================  
// Copyright (c) 2025 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.  
//  
// This software is the confidential and proprietary information of FLIR  
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You  
// shall not disclose such Confidential Information and shall use it only in  
// accordance with the terms of the license agreement you entered into  
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).  
//  
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE  
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE  
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES  
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING  
// THIS SOFTWARE OR ITS DERIVATIVES.  
//  =============================================================================

# README

# TABLE OF CONTENTS

- [1. INSTALLATION]
  - [1.1 WINDOWS]
  - [1.2 LINUX]
  - [1.3 MACOS]
- [2. API DIFFERENCES]
- [3. REMOVE PYSPIN]
- [4. TROUBLESHOOT]
  - [4.1 LINUX ISSUES]


PySpin is a wrapper for Spinnaker library.

Teledyne Machine Vision's website is located at https://www.teledynevisionsolutions.com/solutions/machine-vision/

The `PySpin` Python extension provides a common software interface to control and acquire images from `Teledyne USB 3.0, GigE`, and USB 2.0 cameras using the same API under 32-bit or 64-bit Windows.

=============================================================================
## 1. INSTALLATION
=============================================================================

-----------------------------------------------------------------------------
### 1.1 WINDOWS
-----------------------------------------------------------------------------

1. Install Python.  
Currently we support Python `3.5`, `3.6`, `3.7`, `3.8` and `3.10`.  
   To download Python, visit https://www.python.org/downloads/.  
   Note that the Python website defaults to 32-bit interpreters, so if you want a 64-bit 
   version of Python you have to click into the specific release version.

2. (Optional) Set the PATH environment variable for your Python installation.  
   This may have been done automatically as part of installation, but to do
   this manually you have to open Environment Variables through the following:  
   `My Computer > Properties > Advanced System Settings > Environment Variables`  

   Add your Python installation location to your PATH variable. For example,
   if you installed Python at `C:\Python310\`, you would add the following entry
   to the PATH variable:

       C:\Python310\<rest_of_path>

3. Configure your Python installation. From a command line, run the following
   commands to update and install dependencies for your associated Python version:

       <python version> -m ensurepip
       <python version> -m pip install --upgrade pip numpy matplotlib

   `NumPy` is a requirement for PySpin.  
   For Python version < 3.10 `NumPy` needs to be a 1.x version, 1.15 or above.  
   For Python version = 3.10 `NumPy` needs to be a 1.x version, 1.22 or above.
   `Matplotlib` is not required for the library itself but is used in some of our examples to highlight possible usages of PySpin.  
   For better support of `Matplotlib` output image file formats, `Pillow` is suggested to be installed.  
   Note: some versions of `Pillow` might NOT support some Python versions.

   The full list of supported `Pillow` versions given a Python version can be found here:  
   https://pillow.readthedocs.io/en/stable/installation.html#notes

   For example, with `Python 3.10`, install a supported Pillow using the following command: ex.

       py -3.10 -m pip install Pillow==9.2.0

4. To ensure prerequisites such as drivers and `Visual Studio` redistributables
   are installed on the system, run the `Spinnaker SDK` installer that corresponds
   with the PySpin version number.  
   For example, if installing `PySpin 3.0.0.0`, install `Spinnaker 3.0.0.0` beforehand, selecting only the `Visual Studio` runtimes and drivers.

5. Run the following command to install PySpin to your associated Python version.
   This command assumes you have your PATH variable set correctly for Python:

       <python version> -m pip install spinnaker_python-3.x.x.x-cp3x-cp3x-win_amd64.whl

   Ensure that the wheel downloaded matches the Python version you are installing to!

After installation, `PySpin` examples can be ran directly from the command prompt.  
For example, if `PySpin` is installed for `Python 3.10`, run a preinstalled example using the following: ex.

    py -3.10 Examples\Python3\Acquisition.py

-----------------------------------------------------------------------------
### 1.2 LINUX
-----------------------------------------------------------------------------

1. Check that pip is available for your respective Python versions by running the following command:

       sudo apt-get install python-pip python3-pip

2. Install library dependencies for `PySpin`: `Numpy` and `Matplotlib`.  
   `NumPy` needs to be at least version `1.15` or above for `Ubuntu 20.04` and version 1.22 or above for `Ubuntu 22.04`.
   `Matplotlib` is not required for the library itself but is used in some of our examples to highlight possible usages of `PySpin`.  
   Install these dependencies by running one of the following commands.  

   - Install for Python 3.6, user only:

         python3.6 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.6, site wide:

         sudo python3.6 -m pip install --upgrade numpy matplotlib

   - Install for Python 3.7, user only:

         python3.7 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.7, site wide:

         sudo python3.7 -m pip install --upgrade numpy matplotlib

   - Install for Python 3.8, user only:

         python3.8 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.8, site wide:

         sudo python3.8 -m pip install --upgrade numpy matplotlib

   - Install for Python 3.10, user only:

         python3.10 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.10, site wide:

         sudo python3.10 -m pip install --upgrade numpy matplotlib

   For better support of `Matplotlib` output image file formats, `Pillow` is suggested to be installed.  
   Note: some versions of `Pillow` might NOT support some Python versions.

   The full list of supported `Pillow` versions given a Python version can be found here:  
   https://pillow.readthedocs.io/en/stable/installation.html#notes

   For example, with `Python 3.10`, install a supported `Pillow` using the following command: ex.

       python3.10 -m pip install Pillow==9.2.0

3. Ensure that the corresponding version of the `Spinnaker SDK` Debian packages
   and their prerequisites are installed beforehand  
   (ex. install the 3.0.0.0 packages if the wheel version is also 3.0.0.0)

4. Install wheel for specific Python version. This can be installed site-wide
   for all users or for a specific user.

   - Python 3.6, site wide:

         sudo python3.6 -m pip install spinnaker_python-3.x.x.x-cp36-cp36m-linux_x86_64.whl

   - Python 3.6, user only:

         python3.6 -m pip install --user spinnaker_python-3.x.x.x-cp36-cp36m-linux_x86_64.whl

   - Python 3.7, site wide:

         sudo python3.7 -m pip install spinnaker_python-3.x.x.x-cp37-cp37m-linux_x86_64.whl

   - Python 3.7, user only:

         python3.7 -m pip install --user spinnaker_python-3.x.x.x-cp37-cp37m-linux_x86_64.whl

   - Python 3.8, site wide:

         sudo python3.8 -m pip install spinnaker_python-3.x.x.x-cp38-cp38-linux_x86_64.whl

   - Python 3.8, user only:

         python3.8 -m pip install --user spinnaker_python-3.x.x.x-cp38-cp38-linux_x86_64.whl

   - Python 3.10, site wide:

         sudo python3.10 -m pip install spinnaker_python-3.x.x.x-cp310-cp310-linux_x86_64.whl

   - Python 3.10, user only:

         python3.10 -m pip install --user spinnaker_python-3.x.x.x-cp310-cp310-linux_x86_64.whl

5. The examples are located in the Examples folder of the extracted tarball. Run with:
   ex.
   
       python3.10 Examples/Python3/Acquisition.py

-----------------------------------------------------------------------------
### 1.3 MACOS
-----------------------------------------------------------------------------

1. Check that Python is installed. 

   There are several ways to install Up-to-date Python packages, but the recommended way is to use pyenv - the Python package manager, which manages multiple versions of Python effectively.  
   (installing Python using a method that does not use pyenv, can result in run-time errors due to mixed running Python versions)

   For example: to install the specific Python version 3.7.7 do the following steps:  
   
   - Update brew
  
         brew update

   - Install the pyenv tool
   
         brew install pyenv

   - Install the specific Python version 3.7.7

         pyenv install 3.7.7

   - Set Python version globally.

         pyenv global 3.7.7

   - Adjust the shell's path into the shell (e.g. .zshrc, .bash_profile)
    
         echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile

   - Reset the current shell
      
         source ~/.bash_profile

   - See which versions of Python are installed (e.g. * 3.7.7 (set by ~/.pyenv/version))

         pyenv versions

   - Check the Python version (e.g. Python 3.7.7)

         python3.7 -V

   - Verify that the Python uses the pyenv related path (e.g. ~/.pyenv/shims/python3.7)
   
         which python3.7


2. Update pip for Python. Run the following command for your version of Python:  

       sudo <python version> -m ensurepip

   This will install a version of pip and allow you to update or install new wheels.

3. Install library dependencies for `PySpin`: `Numpy` and `Matplotlib`.  
   `NumPy` is a requirement for `PySpin` and needs to be at least version 1.22 or above.  
   `Matplotlib` is not required for the library itself but is used in some of
   our examples to highlight possible usages of `PySpin`.  
   Install these dependencies by running one of the following commands.

   - Install for Python 3.6, user only:

         python3.6 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.6, site wide:

         sudo python3.6 -m pip install --upgrade numpy matplotlib

   - Install for Python 3.7, user only:

         python3.7 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.7, site wide:

         sudo python3.7 -m pip install --upgrade numpy matplotlib

   - Install for Python 3.8, user only:

         python3.8 -m pip install --upgrade --user numpy matplotlib

   - Install for Python 3.8, site wide:

         sudo python3.8 -m pip install --upgrade numpy matplotlib


  For better support of `Matplotlib` output image file formats, `Pillow` is suggested to be installed.  
  Note: some versions of `Pillow` might NOT support some Python versions.  

   The full list of supported Pillow versions given a Python version can be found here:
   https://pillow.readthedocs.io/en/stable/installation.html#notes

   For example, with Python 3.8, install a supported Pillow using the following command: ex.

       python3.8 -m pip install Pillow==7.0.0

4. Ensure that the corresponding version of the Spinnaker SDK MacOS packages and their 
   prerequisites are installed beforehand.  
   (ex. install 3.0.0.0 packages if the wheel version is also 3.0.0.0)

5. Install the PySpin wheel for specific Python version. ex. for 64-bit Python 3.7

       sudo python3.8 -m pip install spinnaker_python-3.x.x.x-cp37-cp37mu-macos_x86_x64.whl" 

6. The examples are located in the Examples folder of the extracted tarball.  
   Run with: ex.
   
       python3.8 Examples/Python3/Acquisition.py

=============================================================================
## 2. API DIFFERENCES
=============================================================================

Except for the changes listed below, most function names are exactly the same
as the C++ API. See examples for PySpin usage!

- All methods of SpinnakerException no longer exist, please replace all
  usages of SpinnakerException with any of the following attributes:  
    - message: Normal exception message.
    - fullmessage: Exception message including line, file, function,
                   build date, and time (from C++ library).  
    - errorcode: Integer error code of the exception.  
  
  The SpinnakerException instance itself can be printed, as it derives from
  the BaseException class and has a default __str__ representation.  
  See examples for usage.

- Image creation using NumPy arrays (although the int type of the array must be uint8)

- The majority of headers from the C++ API have been wrapped, with the exception of:
    - Headers with "Adapter" or "Port" in the name
    - NodeMapRef.h, NodeMapFactory.h
    - Synch.h, GCSynch.h, Counter.h, filestream.h

- INode and IValue types (esp. returned from GetNode()) have to
  be initialized to their respective pointer types  
  (ex. CFloatPtr, CEnumerationPtr) to access their functions

- CameraPtr, CameraList, InterfacePtr, InterfaceList, and SystemPtr  
  have to be manually released and/or deleted before program exit (use del operator)
    - See EnumerationEvents example

- Image.GetData() returns a 1-D NumPy array of integers, the int type
  depends on the pixel format of the image

- Image.GetNDArray() returns a 2 or 3-D NumPy array of integers, only for select
  image formats.  
  This can be used in libraries such as PIL and/or OpenCV.

- Node callbacks take in a callback class instead of a function pointer
    - Register is now RegisterNodeCallback, Deregister is now DeregisterNodeCallback
    - See NodeMapCallback example for more details

- IImage.CalculateChannelStatistics(StatisticsChannel channel) returns
  a ChannelStatistics object representing stats for the given channel
  in the image.  
  These stats are properties within the ChannelStatistics object.  
  Please see the docstring for details.  
  This replaces ImageStatistics!

- Pass-by-reference functions now return the type and take in void
    - GetFeatures() returns a Python list of IValue, instead of taking
      in a FeatureList_t reference
    - GetChildren() returns a Python list of INode, instead of taking
      in a NodeList_t reference
    - Same with GetEntries(), GetNodes()
    - GetPropertyNames() returns a Python list of str,
      instead of taking in a gcstring_vector reference

- Methods Get() and Set() for IRegister and register nodes use NumPy arrays
    - Get() takes in the length of the register to read and two optional
      bools, returns a NumPy array
    - Set() takes in a single NumPy array

=============================================================================
## 3. REMOVE PYSPIN
=============================================================================

Removing or updating PySpin is similar to removing or updating other wheels.

For Windows, if you need to remove PySpin, the following command needs to be
run from an administrator command prompt to remove your associated Python version:

    <python version> -m pip uninstall spinnaker-python

For Linux or MacOS, if you need to remove PySpin from a user-specific install, run
the following command to remove your associated Python version:

    <python version> -m pip uninstall spinnaker-python

For Linux or MacOS, if you need to remove PySpin from a site-wide install the
following command needs to be run as sudo to remove your associated Python version:

    sudo <python version> -m pip uninstall spinnaker-python


=============================================================================
## 4. TROUBLESHOOT
=============================================================================

-----------------------------------------------------------------------------
### 4.1 Numpy 2.x: Module Compatibility Notice
-----------------------------------------------------------------------------

PySpin does not currently support NumPy 2.x versions. You may need to downgrade
the existing NumPy module to meet the requirement. For example:

```sh
# Downgrade NumPy to version 1.15
<python version> -m pip install --upgrade numpy==1.15
```

-----------------------------------------------------------------------------
### 4.2 Matplotlib: Module Compatibility Notice
-----------------------------------------------------------------------------


To ensure compatibility, you may need to upgrade related modules such as `Matplotlib` to match the 
recommended version for the installed Python version. For example:

```sh
# For Python 3.9, upgrade Matplotlib to version 3.9
<python version> -m pip install --upgrade pip numpy matplotlib
# or pip install --upgrade pip numpy matplotlib==3.9
```

-----------------------------------------------------------------------------
### 4.3 Issue with Numpy 1.19.5 on Linux ARM64 when Importing PySpin
-----------------------------------------------------------------------------

An issue exists with `Numpy` 1.19.5 on the Linux ARM64 architecture, where importing `PySpin` can trigger an "Illegal instruction" error. This problem stems from a bug in `Numpy`, which has been resolved in version 1.20. However, `Numpy` 1.20 and later versions do not support `Python` 3.6 or earlier.

For further details, refer to the Numpy issue discussion.
https://github.com/numpy/numpy/issues/18131#issuecomment-794200556

#### Workarounds:

You can workaround the issue by:
- Downgrading `Numpy` to version 1.19.4
- Upgrading `Numpy` to version 1.20.x (Python 3.7 or later)
- Setting the environment variable 
      ```sh
      OPENBLAS_CORETYPE=ARMV8
      ```
- Compiling from source on the failing ARM hardware

      ```sh
      pip install --no-binary :all: numpy==1.19.5
      ```
