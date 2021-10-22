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
#
# CounterAndTimer.py shows how to setup a Pulse Width Modulation (PWM)
# signal using counters and timers. The camera will output the PWM signal via
# strobe, and capture images at a rate defined by the PWM signal as well.
# Users should take care to use a PWM signal within the camera's max
# frame rate (by default, the PWM signal is set to 50 Hz).
#
# Counter and Timer functionality is only available for BFS and Oryx Cameras.
# For details on the hardware setup, see our kb article, "Using Counter and
# Timer Control"; https://www.flir.com/support-center/iis/machine-vision/application-note/using-counter-and-timer-control

import os
import PySpin
import sys

NUM_IMAGES = 10  # number of images to grab


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                feature_string = node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'
                print('{}: {}'.format(node_feature.GetName(), feature_string))

        else:
            print('Device control information not available.')

        print('')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def setup_counter_and_timer(nodemap):
    """
    This function configures the camera to setup a Pulse Width Modulation signal using
    Counter and Timer functionality.  By default, the PWM signal will be set to run at
    50hz, with a duty cycle of 70%.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('Configuring Pulse Width Modulation signal')

    try:
        result = True

        # Set Counter Selector to Counter 0
        node_counter_selector = PySpin.CEnumerationPtr(nodemap.GetNode('CounterSelector'))

        # Check to see if camera supports Counter and Timer functionality
        if not PySpin.IsAvailable(node_counter_selector):
            print('\nCamera does not support Counter and Timer Functionality.  Aborting...\n')
            return False

        if not PySpin.IsWritable(node_counter_selector):
            print('\nUnable to set Counter Selector (enumeration retrieval). Aborting...\n')
            return False

        entry_counter_0 = node_counter_selector.GetEntryByName('Counter0')
        if not PySpin.IsAvailable(entry_counter_0) or not PySpin.IsReadable(entry_counter_0):
            print('\nUnable to set Counter Selector (entry retrieval). Aborting...\n')
            return False

        counter_0 = entry_counter_0.GetValue()

        node_counter_selector.SetIntValue(counter_0)

        # Set Counter Event Source to MHzTick
        node_counter_event_source = PySpin.CEnumerationPtr(nodemap.GetNode('CounterEventSource'))
        if not PySpin.IsAvailable(node_counter_event_source) or not PySpin.IsWritable(node_counter_event_source):
            print('\nUnable to set Counter Event Source (enumeration retrieval). Aborting...\n')
            return False

        entry_counter_event_source_mhz_tick = node_counter_event_source.GetEntryByName('MHzTick')
        if not PySpin.IsAvailable(entry_counter_event_source_mhz_tick) \
                or not PySpin.IsReadable(entry_counter_event_source_mhz_tick):
            print('\nUnable to set Counter Event Source (entry retrieval). Aborting...\n')
            return False

        counter_event_source_mhz_tick = entry_counter_event_source_mhz_tick.GetValue()

        node_counter_event_source.SetIntValue(counter_event_source_mhz_tick)

        # Set Counter Duration to 14000
        node_counter_duration = PySpin.CIntegerPtr(nodemap.GetNode('CounterDuration'))
        if not PySpin.IsAvailable(node_counter_duration) or not PySpin.IsWritable(node_counter_duration):
            print('\nUnable to set Counter Duration (integer retrieval). Aborting...\n')
            return False

        node_counter_duration.SetValue(14000)

        # Set Counter Delay to 6000
        node_counter_delay = PySpin.CIntegerPtr(nodemap.GetNode('CounterDelay'))
        if not PySpin.IsAvailable(node_counter_delay) or not PySpin.IsWritable(node_counter_delay):
            print('\nUnable to set Counter Delay (integer retrieval). Aborting...\n')
            return False

        node_counter_delay.SetValue(6000)

        # Determine Duty Cycle of PWM signal
        duty_cycle = float(node_counter_duration.GetValue()) / (float(node_counter_duration.GetValue() +
                                                                      node_counter_delay.GetValue())) * 100

        print('\nThe duty cycle has been set to {}%'.format(duty_cycle))

        # Determine pulse rate of PWM signal
        pulse_rate = 1000000 / float(node_counter_duration.GetValue() + node_counter_delay.GetValue())

        print('\nThe pulse rate has been set to {} Hz'.format(pulse_rate))

        # Set Counter Trigger Source to Frame Trigger Wait
        node_counter_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('CounterTriggerSource'))
        if not PySpin.IsAvailable(node_counter_trigger_source) or not PySpin.IsWritable(node_counter_trigger_source):
            print('\nUnable to set Counter Trigger Source (enumeration retrieval). Aborting...\n')
            return False

        entry_counter_trigger_source_ftw = node_counter_trigger_source.GetEntryByName('FrameTriggerWait')
        if not PySpin.IsAvailable(entry_counter_trigger_source_ftw)\
                or not PySpin.IsReadable(entry_counter_trigger_source_ftw):
            print('\nUnable to set Counter Trigger Source (entry retrieval). Aborting...\n')
            return False

        counter_trigger_source_ftw = entry_counter_trigger_source_ftw.GetValue()

        node_counter_trigger_source.SetIntValue(counter_trigger_source_ftw)

        # Set Counter Trigger Activation to Level High
        node_counter_trigger_activation = PySpin.CEnumerationPtr(nodemap.GetNode('CounterTriggerActivation'))
        if not PySpin.IsAvailable(node_counter_trigger_activation) or \
                not PySpin.IsWritable(node_counter_trigger_activation):
            print('\nUnable to set Counter Trigger Activation (enumeration retrieval). Aborting...\n')
            return False

        entry_counter_trigger_source_lh = node_counter_trigger_activation.GetEntryByName('LevelHigh')
        if not PySpin.IsAvailable(entry_counter_trigger_source_lh) \
                or not PySpin.IsReadable(entry_counter_trigger_source_lh):
            print('\nUnable to set Counter Trigger Activation (entry retrieval). Aborting...\n')
            return False

        counter_trigger_level_high = entry_counter_trigger_source_lh.GetValue()

        node_counter_trigger_activation.SetIntValue(counter_trigger_level_high)

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def configure_digital_io(nodemap):
    """
    This function configures the GPIO to output the PWM signal.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('\nConfiguring GPIO strobe output')

    try:
        result = True
        camera_family_bfs = "BFS"
        camera_family_oryx = "ORX"

        # Determine camera family
        node_device_name = PySpin.CStringPtr(nodemap.GetNode('DeviceModelName'))
        if not PySpin.IsAvailable(node_device_name) or not PySpin.IsReadable(node_device_name):
            print('\nUnable to determine camera family. Aborting...\n')
            return False

        camera_model = node_device_name.GetValue()

        # Set Line Selector
        node_line_selector = PySpin.CEnumerationPtr(nodemap.GetNode('LineSelector'))
        if not PySpin.IsAvailable(node_line_selector) or not PySpin.IsWritable(node_line_selector):
            print('\nUnable to set Line Selector (enumeration retrieval). Aborting...\n')
            return False

        if camera_family_bfs in camera_model:

            entry_line_selector_line_1 = node_line_selector.GetEntryByName('Line1')
            if not PySpin.IsAvailable(entry_line_selector_line_1) or not PySpin.IsReadable(entry_line_selector_line_1):
                print('\nUnable to set Line Selector (entry retrieval). Aborting...\n')
                return False

            line_selector_line_1 = entry_line_selector_line_1.GetValue()

            node_line_selector.SetIntValue(line_selector_line_1)

        elif camera_family_oryx in camera_model:

            entry_line_selector_line_2 = node_line_selector.GetEntryByName('Line2')
            if not PySpin.IsAvailable(entry_line_selector_line_2) or not PySpin.IsReadable(entry_line_selector_line_2):
                print('\nUnable to set Line Selector (entry retrieval). Aborting...\n')
                return False

            line_selector_line_2 = entry_line_selector_line_2.GetValue()

            node_line_selector.SetIntValue(line_selector_line_2)

            # Set Line Mode to output
            node_line_mode = PySpin.CEnumerationPtr(nodemap.GetNode('LineMode'))
            if not PySpin.IsAvailable(node_line_mode) or not PySpin.IsWritable(node_line_mode):
                print('\nUnable to set Line Mode (enumeration retrieval). Aborting...\n')
                return False

            entry_line_mode_output = node_line_mode.GetEntryByName('Output')
            if not PySpin.IsAvailable(entry_line_mode_output) or not PySpin.IsReadable(entry_line_mode_output):
                print('\nUnable to set Line Mode (entry retrieval). Aborting...\n')
                return False

            line_mode_output = entry_line_mode_output.GetValue()

            node_line_mode.SetIntValue(line_mode_output)

        # Set Line Source for Selected Line to Counter 0 Active
        node_line_source = PySpin.CEnumerationPtr(nodemap.GetNode('LineSource'))
        if not PySpin.IsAvailable(node_line_source) or not PySpin.IsWritable(node_line_source):
            print('\nUnable to set Line Source (enumeration retrieval). Aborting...\n')
            return False

        entry_line_source_counter_0_active = node_line_source.GetEntryByName('Counter0Active')
        if not PySpin.IsAvailable(entry_line_source_counter_0_active) \
                or not PySpin.IsReadable(entry_line_source_counter_0_active):
            print('\nUnable to set Line Source (entry retrieval). Aborting...\n')
            return False

        line_source_counter_0_active = entry_line_source_counter_0_active.GetValue()

        node_line_source.SetIntValue(line_source_counter_0_active)

        if camera_family_bfs in camera_model:
            # Change Line Selector to Line 2 and Enable 3.3 Voltage Rail
            entry_line_selector_line_2 = node_line_selector.GetEntryByName('Line2')
            if not PySpin.IsAvailable(entry_line_selector_line_2) or not PySpin.IsReadable(entry_line_selector_line_2):
                print('\nUnable to set Line Selector (entry retrieval). Aborting...\n')
                return False

            line_selector_line_2 = entry_line_selector_line_2.GetValue()

            node_line_selector.SetIntValue(line_selector_line_2)

            node_voltage_enable = PySpin.CBooleanPtr(nodemap.GetNode('V3_3Enable'))
            if not PySpin.IsAvailable(node_voltage_enable) or not PySpin.IsWritable(node_voltage_enable):
                print('\nUnable to set Voltage Enable (boolean retrieval). Aborting...\n')
                return False

            node_voltage_enable.SetValue(True)

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def configure_exposure_and_trigger(nodemap):
    """
    This function configures the camera to set a manual exposure value and enables
    camera to be triggered by the PWM signal.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('\nConfiguring Exposure and Trigger')

    try:
        result = True

        # Turn off auto exposure
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
            print('\nUnable to set Exposure Auto (enumeration retrieval). Aborting...\n')
            return False

        entry_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(entry_exposure_auto_off) or not PySpin.IsReadable(entry_exposure_auto_off):
            print('\nUnable to set Exposure Auto (entry retrieval). Aborting...\n')
            return False

        exposure_auto_off = entry_exposure_auto_off.GetValue()

        node_exposure_auto.SetIntValue(exposure_auto_off)

        # Set Exposure Time to less than 1/50th of a second (5000 us is used as an example)
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            print('\nUnable to set Exposure Time (float retrieval). Aborting...\n')
            return False

        node_exposure_time.SetValue(5000)

        # Ensure trigger mode is off
        #
        # *** NOTES ***
        # The trigger must be disabled in order to configure
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
            print('\nUnable to disable trigger mode (node retrieval). Aborting...\n')
            return False

        entry_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(entry_trigger_mode_off) or not PySpin.IsReadable(entry_trigger_mode_off):
            print('\nUnable to disable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        node_trigger_mode.SetIntValue(entry_trigger_mode_off.GetValue())

        # Set Trigger Source to Counter 0 Start
        node_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerSource'))
        if not PySpin.IsAvailable(node_trigger_source) or not PySpin.IsWritable(node_trigger_source):
            print('\nUnable to set trigger source (enumeration retrieval). Aborting...\n')
            return False

        entry_trigger_source_counter_0_start = node_trigger_source.GetEntryByName('Counter0Start')
        if not PySpin.IsAvailable(entry_trigger_source_counter_0_start)\
                or not PySpin.IsReadable(entry_trigger_source_counter_0_start):
            print('\nUnable to set trigger mode (enum entry retrieval). Aborting...\n')
            return False

        node_trigger_source.SetIntValue(entry_trigger_source_counter_0_start.GetValue())

        # Set Trigger Overlap to Readout
        node_trigger_overlap = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerOverlap'))
        if not PySpin.IsAvailable(node_trigger_overlap) or not PySpin.IsWritable(node_trigger_overlap):
            print('\nUnable to set Trigger Overlap (enumeration retrieval). Aborting...\n')
            return False

        entry_trigger_overlap_ro = node_trigger_overlap.GetEntryByName('ReadOut')
        if not PySpin.IsAvailable(entry_trigger_overlap_ro) or not PySpin.IsReadable(entry_trigger_overlap_ro):
            print('\nUnable to set Trigger Overlap (entry retrieval). Aborting...\n')
            return False

        trigger_overlap_ro = entry_trigger_overlap_ro.GetValue()

        node_trigger_overlap.SetIntValue(trigger_overlap_ro)

        # Turn trigger mode on
        entry_trigger_mode_on = node_trigger_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(entry_trigger_mode_on) or not PySpin.IsReadable(entry_trigger_mode_on):
            print('\nUnable to enable trigger mode (enum entry retrieval). Aborting...\n')
            return False

        node_trigger_mode.SetIntValue(entry_trigger_mode_on.GetValue())

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function acquires and saves 10 images from a device; please see
    Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('\n*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enumeration retrieval). Aborting...\n')
            return False

        entry_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(entry_acquisition_mode_continuous)\
                or not PySpin.IsReadable(entry_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (enum entry retrieval). Aborting...\n')
            return False

        acquisition_mode_continuous = entry_acquisition_mode_continuous.GetValue()

        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as {}...'.format(device_serial_number))

        print('')

        # Retrieve, convert, and save images
        for i in range(NUM_IMAGES):
            try:

                #  Retrieve next received image and ensure image completion
                image_result = cam.GetNextImage(1000)

                if image_result.IsIncomplete():
                    print('Image incomplete with image status {} ...'.format(image_result.GetImageStatus()))

                else:

                    #  Print image information; height and width recorded in pixels
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed image {}, width = {}, height = {}'.format(i, width, height))

                    #  Convert image to mono 8
                    image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                    # Create a unique filename
                    if device_serial_number:
                        filename = 'CounterAndTimer-{}-{}.jpg'.format(device_serial_number, i)
                    else:  # if serial number is empty
                        filename = 'CounterAndTimer-{}.jpg'.format(i)

                    #  Save image
                    image_converted.Save(filename)
                    print('Image saved at {}'.format(filename))

                    #  Release image
                    image_result.Release()
                    print('')

            except PySpin.SpinnakerException as ex:
                print('Error: {}'.format(ex))
                return False

        #  End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def reset_trigger(nodemap):
    """
    This function returns the camera to a normal state by turning off trigger mode.

    *** NOTES ***
    This function turns off trigger mode, but does not change the trigger source.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Turn trigger mode back off
        node_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode('TriggerMode'))
        if not PySpin.IsAvailable(node_trigger_mode) or not PySpin.IsWritable(node_trigger_mode):
            print('Unable to disable trigger mode (node retrieval). Non-fatal error...\n')

        entry_trigger_mode_off = node_trigger_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(entry_trigger_mode_off) or not PySpin.IsReadable(entry_trigger_mode_off):
            print('Unable to disable trigger mode (enum entry retrieval). Non-fatal error...\n')

        node_trigger_mode.SetIntValue(entry_trigger_mode_off.GetValue())

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        return False

    return result


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see the NodeMapInfo example
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

        # Configure Counter and Timer setup
        result &= setup_counter_and_timer(nodemap)
        if not result:
            return result

        # Configure DigitalIO (GPIO output)
        result &= configure_digital_io(nodemap)
        if not result:
            return result

        # Configure Exposure and Trigger
        result &= configure_exposure_and_trigger(nodemap)
        if not result:
            return result

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Reset trigger
        result &= reset_trigger(nodemap)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
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
    print('Library version: {}.{}.{}.{}'.format(version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: {}'.format(num_cameras))

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

        print('Running example for camera {}...'.format(i))

        result &= run_single_camera(cam)
        print('Camera {} example complete... \n'.format(i))

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
