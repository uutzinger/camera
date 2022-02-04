#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This is a python install script written for camera python package.

import io
import os
import sys

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(

    name='camera-util',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # http://packaging.python.org/en/latest/tutorial.html#version
    version='1.0.6',

    description=("Python wrapper for USB, blackfly, Raspi, Jetson Nano cameras"  
                 "Works on Windows, Raspian, JetsonNano, MacOS"   ),

    # The project's main homepage.
    url='https://github.com/uutzinger/camera',

    #use_scm_version={
    #    # This is needed for the PyPI version munging in the Github Actions release.yml
    #    "git_describe_command": "git describe --tags --long",
    #    "local_scheme": "no-local-version",
    #},
    #setup_requires=["setuptools_scm"],
    long_description=long_description,
    long_description_content_type="text/x-rst",

    # Author details
    author='Urs Utzinger',
    author_email='uutzinger@gmail.com',

    #install_requires=[
    #    #'opencv-python',
    #    'h5py',
    #    'tifffile',
    #    'numpy'
    #],

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Video :: Capture',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='camera, driver, opencv, blackfly',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=["camera", "camera.capture", "camera.streamer", "camera.processor"],
)