# coding=utf-8
# =============================================================================
# Copyright (c) 2025 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
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
#
# StereoGPIO.py shows how to set the GPIO of the stereo camera.
#
# This example demonstrates how to set the GPIO of the camera.
# After setting the GPIO controls, images are grabbed pending on a signal
# in the GPIO line.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import os
import PySpin
import sys
import platform

def validate_image_list(stream_transmit_flags, image_list):
    """
    This function validates the image list.

    :param stream_transmit_flags: list of boolean flags refering to the various streams indicating if to trasmit the stream.
    :param image_list: list of aquired streams.
    :type stream_transmit_flags: dict
    :type image_list: ImageList
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    
    is_image_list_incomplete = False;

    if stream_transmit_flags['rawSensor1TransmitEnabled']:
        image = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RAW_SENSOR1);
        is_image_list_incomplete |= ((image is None) or image.IsIncomplete())

    if stream_transmit_flags['rawSensor2TransmitEnabled']:
        image = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RAW_SENSOR2);
        is_image_list_incomplete |= ((image is None) or image.IsIncomplete())

    if stream_transmit_flags['rectSensor1TransmitEnabled']:
        image = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RECTIFIED_SENSOR1);
        is_image_list_incomplete |= ((image is None) or image.IsIncomplete())

    if stream_transmit_flags['rectSensor2TransmitEnabled']:
        image = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RECTIFIED_SENSOR2);
        is_image_list_incomplete |= ((image is None) or image.IsIncomplete())

    if stream_transmit_flags['disparityTransmitEnabled']:
        image = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_DISPARITY_SENSOR1);
        is_image_list_incomplete |= ((image is None) or image.IsIncomplete())

    if is_image_list_incomplete:
        print('Image List is incomplete.')

    return not is_image_list_incomplete
   
def configure_gpio(cam):
    """
    This function configures the camera to use a hardware trigger via the GPIO.
    First, trigger mode is set to off in order to select the trigger source. 
    Then input line 0 is selected to trigger frame start.
    Also, output line 1 is set to to trigger on exposure active.
    Once the trigger source has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the chosen trigger.

     :param cam: Camera to configure trigger for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    result = True

    print('*** CONFIGURING TRIGGER ***\n')

    try:
        nodemap = cam.GetNodeMap()

        # Set triggerMode to Off
        # Trigger mode must be off when setting the trigger source.
        if not set_trigger(nodemap, False):
            print('Unable to disable the trigger mode. Aborting...')
            return False

        # Set trigger source to line 0
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsReadable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print('Unable to get or set the trigger source (node retrieval). Aborting...')
            return False

        node_trigger_source_hardware = node_trigger_source.GetEntryByName('Line0')
        if not PySpin.IsReadable(node_trigger_source_hardware):
            print('Unable to get trigger source (Line0 enum entry retrieval). Aborting...')
            return False
        node_trigger_source.SetIntValue(node_trigger_source_hardware.GetValue())
        print('Trigger source set to hardware, line 0.')

        # Set trigger selector to frame start
        # For this example, the trigger selector should be set to frame start.
        # This is the default for most cameras.
        node_trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSelector'))
        if not PySpin.IsReadable(node_trigger_selector) or not PySpin.IsWritable(node_trigger_selector):
            print('Unable to get or set the trigger selector (node retrieval). Aborting...')
            return False

        node_trigger_selector_framestart = node_trigger_selector.GetEntryByName('FrameStart')
        if not PySpin.IsReadable(node_trigger_selector_framestart):
            print('Unable to get the trigger selector (enum entry retrieval). Aborting...')
            return False
        node_trigger_selector.SetIntValue(node_trigger_selector_framestart.GetValue())

        print('Trigger selector set to frame start.')

        # Set line selector to line 1
        node_line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        if not PySpin.IsReadable(node_line_selector) or not PySpin.IsWritable(node_line_selector):
            print('Unable to get or set the trigger selector (node retrieval). Aborting...')
            return False

        node_line_selector_line1 = node_line_selector.GetEntryByName('Line1')
        if not PySpin.IsReadable(node_line_selector_line1):
            print('Unable to get the trigger selector (enum entry retrieval). Aborting...')
            return False
        node_line_selector.SetIntValue(node_line_selector_line1.GetValue())

        print('Line selector set line 1.')

        # Set line source to exposure active
        node_line_source = PySpin.CEnumerationPtr(nodemap.GetNode('LineSource'))
        if not PySpin.IsReadable(node_line_source) or not PySpin.IsWritable(node_line_source):
            print('Unable to get or set the line source (node retrieval). Aborting...')
            return False

        node_line_source_exposure_active = node_line_source.GetEntryByName('ExposureActive')
        if not PySpin.IsReadable(node_line_source_exposure_active):
            print('Unable to get the line source (enum entry retrieval). Aborting...')
            return False
        node_line_source.SetIntValue(node_line_source_exposure_active.GetValue())

        print('Line source set to exposure active.')

        # Set triggerMode to On
        if not set_trigger(nodemap, True):
            print('Unable to enable the trigger mode. Aborting...')
            return False

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def set_trigger(nodemap, is_on):
    """
    This function sets the trigger mode to on/off.

    :param nodemap: Transport layer device nodemap.
    :param is_on: the state to put the trigger mode to.
    :type nodemap: INodeMap
    :type is_on: bool
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    
    stateStr = "On" if is_on else "Off"
    
    try:
        result = True
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsReadable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
            print(f"Unable to update trigger mode (node retrieval). Aborting...")    
            return False

        node_trigger_mode_state = node_trigger_mode.GetEntryByName(stateStr)
        if not PySpin.IsReadable(node_trigger_mode_state):
            print(f"Unable to update trigger mode (enum entry retrieval). Aborting...")    
            return False

        node_trigger_mode.SetIntValue(node_trigger_mode_state.GetValue())

        if is_on:
            print(f"Trigger mode enabled...")
        else:
            print(f"Trigger mode disabled...")

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function acquires and saves multiple image sets from a device.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous

        # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        cam.BeginAcquisition()
           
        stream_transmit_flags = {}
        stream_transmit_flags['rawSensor1TransmitEnabled'] = False
        stream_transmit_flags['rawSensor2TransmitEnabled'] = False
        stream_transmit_flags['rectSensor1TransmitEnabled'] = True
        stream_transmit_flags['rectSensor2TransmitEnabled'] = False
        stream_transmit_flags['disparityTransmitEnabled'] = True

        timeoutInMilliSecs = 5000
        num_images_sets = 3
        if timeoutInMilliSecs == PySpin.EVENT_TIMEOUT_INFINITE:
            print ('\nAcquiring %d image sets pending on GPIO signal, within an infinite time limit.' % num_images_sets)
        else:
            print ('\nAcquiring %d image sets pending on GPIO signal, within a time limit of %f secs.' % (num_images_sets, timeoutInMilliSecs / 1000))
        
        # Retrieve, process the images
        for i in range(num_images_sets):
            try:
                print('Acquiring stereo image set: %d' % i)

                image_list = cam.GetNextImageSync(timeoutInMilliSecs)
                if not validate_image_list(stream_transmit_flags, image_list):
                    print('Failed to get next image set.')
                    continue;

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                result = False
                
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        is_stereo_camera = PySpin.ImageUtilityStereo.IsStereoCamera(cam)
        if is_stereo_camera:
            print('Camera %s is a stereo camera' % device_serial_number)
        else:
            print('Camera %s is not a valid BX camera. Skipping...' % device_serial_number)
            return True

        print('Configuring GPIO...')
        if not configure_gpio(cam):
            print('Failed to configure the GPIO. Aborting...')
            return False

        # Acquire images
        print('Acquiring images...')
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Set trigger to Off
        result |= set_trigger(nodemap, False)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print('Running example for camera %d...' % i)

        result &= run_single_camera(cam)
        print('Camera %d example complete... \n' % i)

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)