# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2021 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================

# LookupTable.py
#
# LookupTable.py shows how to configure lookup tables on the camera.
# It relies on information provided in the Enumeration, Acquisition, and
# NodeMapInfo examples.
#
# It can also be helpful to familiarize yourself with the ImageFormatControl
# and Exposure examples. As they are somewhat shorter and simpler, either
# provides a strong introduction to camera customization.
#
# Lookup tables allow for the customization and control of individual pixels.
# This can be a very powerful and deeply useful tool; however, because use
# cases are context dependent, this example only explores lookup table
# configuration.

import os
import PySpin
import sys

NUM_IMAGES = 10  # number of images to grab


def print_retrieve_node_failure(node, name):
    """"
    This function handles the error prints when a node or entry is unavailable or
    not readable on the connected camera.

    :param node: Node type. "Node" or "Entry"
    :param name: Node name.
    :type node: String
    :type name: String
    :rtype: None
    """
    print("Unable to get {} ({} {} retrieval failed.)".format(node, name, node))
    print("The {} may not be available on all camera models...".format(node))
    print("Please try a Blackfly S camera.")


def configure_lookup_tables(nodemap):
    """
    This function configures lookup tables linearly. This involves selecting the
    type of lookup table, finding the appropriate increment calculated from the
    maximum value, and enabling lookup tables on the camera.

    :param nodemap: Device nodemap
    :type nodemap: INodeMap
    :return: returns True if successful, False otherwise
    :rtype: bool
    """
    result = True
    print("***CONFIGURING LOOKUP TABLES***\n")

    # Select lookup table type
    #
    # ***NOTES ***
    # Setting the lookup table selector. It is important to note that this
    # does not enable lookup tables.

    try:
        lut_selector = PySpin.CEnumerationPtr(nodemap.GetNode("LUTSelector"))
        if not PySpin.IsAvailable(lut_selector) or not PySpin.IsWritable(lut_selector):
            print_retrieve_node_failure("node", "LUTSelector")
            return False

        lut_selector_lut1 = lut_selector.GetEntryByName("LUT1")
        if not PySpin.IsAvailable(lut_selector_lut1) or not PySpin.IsReadable(lut_selector_lut1):
            print_retrieve_node_failure("entry", "LUTSelector LUT1")
            return False

        lut_selector.SetIntValue(lut_selector_lut1.GetValue())
        print("Lookup table selector set to LUT 1...\n")

        # Determine pixel increment and set indexes and values as desired
        #
        # *** NOTES ***
        # To get the pixel increment, the maximum range of the value node must
        # first be retrieved. The value node represents an index, so its value
        # should be one less than a power of 2 (e.g. 511, 1023, etc.). Add 1 to
        # this index to get the maximum range. Divide the maximum range by 512
        # to calculate the pixel increment.
        #
        # Finally, all values (in the value node) and their corresponding
        # indexes (in the index node) need to be set. The goal of this example
        # is to set the lookup table linearly. As such, the slope of the values
        # should be set according to the increment, but the slope of the
        # indexes is inconsequential.

        # Retrieve value node
        lut_value = PySpin.CIntegerPtr(nodemap.GetNode("LUTValue"))
        if not PySpin.IsAvailable(lut_value) or not PySpin.IsWritable(lut_value):
            print_retrieve_node_failure("node", "LUTValue")
            return False

        # Retrieve maximum range
        max_range = lut_value.GetMax() + 1
        print("\tMaximum Range: {}".format(max_range))

        # Calculate increment
        increment = max_range / 512
        print("\tIncrement: {}".format(increment))

        # Retrieve index node
        lut_index = PySpin.CIntegerPtr(nodemap.GetNode("LUTIndex"))
        if not PySpin.IsAvailable(lut_index) or not PySpin.IsWritable(lut_index):
            print_retrieve_node_failure("node", "LUTIndex")
            return False

        # Set values and indexes
        i = 0
        while i < max_range:
            lut_index.SetValue(int(i))
            lut_value.SetValue(int(i))
            i += increment

        print("All lookup table values set...\n")

        # Enable lookup tables
        #
        # *** NOTES ***
        # Once lookup tables have been configured, don"t forget to enable them
        # with the appropriate node.
        #
        # *** LATER ***
        # Once the images with lookup tables have been collected, turn the
        # feature off with the same node.

        lut_enable = PySpin.CBooleanPtr(nodemap.GetNode("LUTEnable"))
        if not PySpin.IsAvailable(lut_enable) or not PySpin.IsWritable(lut_enable):
            print_retrieve_node_failure("node", "LUTEnable")
            return False

        lut_enable.SetValue(True)
        print("Lookup tables enabled...\n")

    except PySpin.SpinnakerException as ex:
        print("Error: {}".format(ex))
        result = False

    return result


def reset_lookup_tables(nodemap):
    """
    This function resets the camera by disabling lookup tables.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: returns True if successful, False otherwise
    :rtype: bool
    """
    result = True

    # Disable lookup tables
    #
    # *** NOTES ***
    # Turn lookup tables off when they are not needed to reduce overhead

    try:
        lut_enable = PySpin.CBooleanPtr(nodemap.GetNode("LUTEnable"))
        if not PySpin.IsAvailable(lut_enable) or not PySpin.IsWritable(lut_enable):
            print("Unable to disable lookup tables. Non-fatal error...\n")
            return False

        lut_enable.SetValue(False)
        print("Lookup tables disabled...\n")

    except PySpin.SpinnakerException as ex:
        print("Error: {}".format(ex))
        result = False

    return result


def print_device_info(nodemap):
    """
    # This function prints the device information of the camera from the transport
    # layer; please see NodeMapInfo example for more in-depth comments on printing
    # device information from the nodemap.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: returns True if successful, False otherwise
    :rtype: bool
    """
    result = True
    print("*** DEVICE INFORMATION ***\n")

    try:
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode("DeviceInformation"))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                if PySpin.IsReadable(node_feature):
                    feature_string = node_feature.ToString()
                else:
                    feature_string = "Node not readable"

                print("{}: {}".format(node_feature.GetName(), feature_string))

        else:
            print("Device control information not available.")

    except PySpin.SpinnakerException as ex:
        print("Error: {}".format(ex))
        result = False

    return result


def acquire_images(cam, nodemap, nodemap_tl_device):
    """
    This function acquires and saves 10 images from a device; please see
    Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from
    :param nodemap: Device nodemap
    :param nodemap_tl_device: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tl_device: INodeMap
    :return: returns True if successful, False otherwise
    :rtype: bool
    """
    result = True
    print("*** IMAGE ACQUISITION ***\n")

    # Set acquisition mode to continuous
    try:
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print("Unable to set acquisition mode to continuous (node retrieval). Aborting...\n")
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or \
                not PySpin.IsReadable(node_acquisition_mode_continuous):
            print("Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting...\n")
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        print("Acquisition mode set to continuous...\n")

        #  Begin acquiring images
        cam.BeginAcquisition()
        print("Acquiring images...\n")

        #  Retrieve device serial number for filename
        device_serial_number = ""
        node_device_serial_number = PySpin.CStringPtr(nodemap_tl_device.GetNode("DeviceSerialNumber"))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print("Device serial number retrieved as {}...".format(device_serial_number))

        print("")

        # Retrieve, convert, and save images
        for i in range(NUM_IMAGES):
            try:

                # Retrieve next received image and ensure image completion
                image_result = cam.GetNextImage(1000)

                if image_result.IsIncomplete():
                    print("Image incomplete with image status {}...".format(image_result.GetImageStatus()))

                else:
                    #  Print image information
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print("Grabbed image {}, width = {}, height = {}".format(i, width, height))

                    #  Convert image to mono 8
                    image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                    # Create a unique filename
                    if device_serial_number:
                        filename = "LookupTable-{}-{}.jpg".format(device_serial_number, i)
                    else:  # if serial number is empty
                        filename = "LookupTable-{}.jpg".format(i)

                    #  Save image
                    image_converted.Save(filename)
                    print("Image saved at {}".format(filename))

                    #  Release image
                    image_result.Release()
                    print("")

            except PySpin.SpinnakerException as ex:
                print("Error: {}".format(ex))
                result = False

        #  End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print("Error: {}".format(ex))
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: returns True if successful, False otherwise
    :rtype: bool
    """
    result = True

    try:
        # Retrieve TL device nodemap and print device information
        nodemap_tl_device = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tl_device)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure lookup tables
        result &= configure_lookup_tables(nodemap)
        if not result:
            return result

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tl_device)

        # Reset lookup tables
        result &= reset_lookup_tables(nodemap)

        # Deinitialize camera
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print("Error: {}".format(ex))
        result = False

    return result


def main():
    """
    Since this application saves images in the current folder
    we must ensure that we have permission to write to this folder.
    If we do not have permission, fail right away.

    :return: returns True if successful, False otherwise
    :rtype: bool
    """
    try:
        test_file = open("test.txt", "w+")
    except IOError:
        print("Unable to write to current directory. Please check permissions.\n")
        input("Press Enter to exit...")
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print("Library version: {}.{}.{}.{}\n".format(version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print("Number of cameras detected: {}\n".format(num_cameras))

    # Finish if there are no cameras
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()
        # Release system instance
        system.ReleaseInstance()
        print("Not enough cameras!\n")
        input("Done! Press Enter to exit...")
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):
        print("Running example for camera {}...\n".format(i))

        result &= run_single_camera(cam)
        print("Camera {} example complete...\n".format(i))

    # Release reference to camera
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input("Done! Press Enter to exit...")
    return result


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
