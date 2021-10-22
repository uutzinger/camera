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

#  BufferHandling.py shows how the different buffer handling modes work.
#  It relies on information provided in the Acquisition and Trigger examples.
#
#  Buffer handling determines the ordering at which images are retrieved, and
#  what occurs when an image is transmitted while the buffer is full.  There are
#  four different buffer handling modes available; NewestFirst, NewestOnly,
#  OldestFirst and OldestFirstOverwrite.
#
#  This example explores retrieving images in a set pattern; triggering the camera
#  while not retrieving an image (letting the buffer fill up), and retrieving
#  images while not triggering.  We cycle through the different buffer handling
#  modes to see which images are retrieved, confirming their identites via their
#  Frame ID values.

import os
import PySpin
import time
import sys

# Total number of buffers
NUM_BUFFERS = 3
# Number of triggers
NUM_TRIGGERS = 6
# Total number of loops
NUM_LOOPS = 9

def configure_trigger(nodemap):
    """
    This function configures the camera to use a trigger. First, trigger mode is
    set to off in order to select the trigger source. Once the trigger source
    has been selected, trigger mode is then enabled, which has the camera
    capture only a single image upon the execution of the trigger.

    :param nodemap: Device nodemap to retrieve images from.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise
    :rtype: bool
    """
    try:
        result = True
        print('\n*** CONFIGURING TRIGGER ***\n')

        # Ensure trigger mode off
        #
        # *** NOTES ***
        # The trigger must be disabled in order to configure the
        # trigger source.
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(trigger_mode) or not PySpin.IsWritable(trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Aborting...\n')
            return False

        trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
        if not PySpin.IsAvailable(trigger_mode_off) or not PySpin.IsReadable(trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        trigger_mode.SetIntValue(trigger_mode_off.GetValue())
        print('Trigger mode disabled...')

        # Set trigger source to software
        trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(trigger_source) or not PySpin.IsWritable(trigger_source):
            print('Unable to set trigger mode (node retrieval). Aborting...')
            return False

        trigger_source_software = PySpin.CEnumEntryPtr(trigger_source.GetEntryByName('Software'))
        if not PySpin.IsAvailable(trigger_source_software) or not PySpin.IsReadable(trigger_source_software):
            print('Unable to set trigger mode (enum entry retrieval). Aborting...')
            return False

        trigger_source.SetIntValue(trigger_source_software.GetValue())
        print('Trigger source set to software...')

        # Turn trigger mode on
        trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('On'))
        if not PySpin.IsAvailable(trigger_mode_on) or not PySpin.IsReadable(trigger_mode_on):
            print('Unable to enable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        trigger_mode.SetIntValue(trigger_mode_on.GetValue())
        print('Trigger mode turned back on...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def grab_next_image_by_trigger(nodemap):
    """
    This function retrieves a single image using the trigger. In this example,
    only a single image is captured and made available for acquisition - as such,
    attempting to acquire two images for a single trigger execution would cause
    the example to hang. This is different from other examples, whereby a
    constant stream of images are being captured and made available for image
    acquisition.

    :param nodemap: Device nodemap to retrieve images from.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise
    :rtype: bool
    """
    try:
        result = True

        # Execute software trigger
        software_trigger_command = PySpin.CCommandPtr(nodemap.GetNode('TriggerSoftware'))
        if not PySpin.IsAvailable(software_trigger_command) or not PySpin.IsWritable(software_trigger_command):
            print('Unable to execute trigger. Aborting...\n')
            return False

        software_trigger_command.Execute()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def reset_trigger(nodemap):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    :param nodemap: Device nodemap to retrieve images from.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise
    :rtype: bool
    """
    try:
        result = True

        # Turn trigger mode back off
        #
        # *** NOTES ***
        # Once all images have been captured, turn trigger mode back off to
        # restore the camera to a clean state.
        trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(trigger_mode) or not PySpin.IsWritable(trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Non-fatal error...\n')
            return False

        trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
        if not PySpin.IsAvailable(trigger_mode_off) or not PySpin.IsReadable(trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Non-fatal error...\n')
            return False

        trigger_mode.SetIntValue(trigger_mode_off.GetValue())
        print('Trigger mode disabled...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap from camera.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        print('\n*** DEVICE INFORMATION ***\n')

        # Retrieve and display Device Information
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function cycles through the four different buffer handling modes.
    It saves three images for three of the buffer handling modes
    (NewestFirst, OldestFirst, and OldestFirstOverwrite).  For NewestOnly,
    it saves one image.

    :param cam: Camera instance to grab images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        print('\n*** IMAGE ACQUISITION ***\n')

        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (node retrieval). Aborting...')
            return False

        # Retrieve entry node from enumeration mode
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve Stream Parameters device nodemap
        s_node_map = cam.GetTLStreamNodeMap()

        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False

        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        if not PySpin.IsAvailable(stream_buffer_count_mode) or not PySpin.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not PySpin.IsAvailable(stream_buffer_count_mode_manual) or not PySpin.IsReadable(stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print('Stream Buffer Count Mode set to manual...')

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

        # Display Buffer Info
        print('\nDefault Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        print('Default Buffer Count: %d' % buffer_count.GetValue())
        print('Maximum Buffer Count: %d' % buffer_count.GetMax())

        buffer_count.SetValue(NUM_BUFFERS)

        print('Buffer count now set to: %d' % buffer_count.GetValue())
        print('\nCamera will be triggered %d times in a row before %d images will be retrieved' % (NUM_TRIGGERS,(NUM_LOOPS-NUM_TRIGGERS)))

        for x in range (0, 4):
            if x == 0:
                handling_mode_entry = handling_mode.GetEntryByName('NewestFirst')
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())
            elif x == 1:
                handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())
            elif x == 2:
                handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())
            elif x == 3:
                handling_mode_entry = handling_mode.GetEntryByName('OldestFirstOverwrite')
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())

            # Begin capturing images
            cam.BeginAcquisition()

            # Sleep for one second; only necessary when using non-BFS/ORX cameras on startup
            if x == 0:
                time.sleep(1)

            try:
                # Software Trigger the camera then  save images
                for loop_cnt in range (0, NUM_LOOPS):
                    if loop_cnt < NUM_TRIGGERS:
                        # Retrieve the next image from the trigger
                        result &= grab_next_image_by_trigger(nodemap)
                        print('\nCamera triggered. No image grabbed')
                    else:
                        print('\nNo trigger. Grabbing image %d' %(loop_cnt-NUM_TRIGGERS))
                        result_image = cam.GetNextImage(500)

                        if result_image.IsIncomplete():
                            print('Image incomplete with image status %s ...\n' % result_image.GetImageStatus())

                    if loop_cnt >= NUM_TRIGGERS:
                        # Retrieve Frame ID
                        print('Frame ID: %d' % result_image.GetFrameID())

                        # Create a unique filename
                        if device_serial_number:
                            filename = '%s-%s-%d.jpg' % (handling_mode_entry.GetSymbolic(),device_serial_number, (loop_cnt-NUM_TRIGGERS))
                        else:
                            filename = '%s-%d.jpg' % (handling_mode_entry.GetSymbolic(),(loop_cnt-NUM_TRIGGERS))

                        # Save image
                        result_image.Save(filename)
                        print('Image saved at %s' % filename)

                        # Release image
                        result_image.Release()

                    # To control the framerate, have the application pause for 250ms.
                    time.sleep(0.25)

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                if handling_mode_entry.GetSymbolic() == 'NewestOnly':
                    print('Error should occur when grabbing image 1 with handling mode set to NewestOnly')
                result = False

            # End acquisition
            cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to acquire images from.
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

        # Configure chunk data
        if configure_trigger(nodemap) is False:
            return False

        # Acquire images and display chunk data
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Reset trigger
        result &= reset_trigger(nodemap)

        # De-initialize camera
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

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

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
        print('\n\nRunning example for camera %d...' % i)

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



