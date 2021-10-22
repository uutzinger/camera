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
# Sequencer.py shows how to use the sequencer to grab images with
# various settings. It relies on information provided in the Enumeration,
# Acquisition, and NodeMapInfo examples.
#
# It can also be helpful to familiarize yourself with the ImageFormatControl
# and Exposure examples as these examples provide a strong introduction to
# camera customization.
#
# The sequencer is another very powerful tool, which can be used to create
# and store multiple states of customized image settings. A very useful
# application of the sequencer is creating high dynamic range images.
#
# This example is probably the most complex and definitely the longest. As
# such, the configuration has been split between three functions. The first
# prepares the camera to set the sequences, the second sets the settings for
# a single state (it is run five times), and the third configures the
# camera to use the sequencer when it acquires images.

import os
import PySpin
import sys

NUM_IMAGES = 10  # number of images to grab


def print_retrieve_node_failure(node, name):
    """"
    This function handles the error prints when a node or entry is unavailable or
    not readable on the connected camera.

    :param node: Node type. "Node' or 'Entry'
    :param name: Node name.
    :type node: String
    :type name: String
    :rtype: None
    """
    print('Unable to get {} ({} {} retrieval failed.)'.format(node, name, node))
    print('The {} may not be available on all camera models...'.format(node))
    print('Please try a Blackfly S camera.')


def configure_sequencer_part_one(nodemap):
    """"
    This function prepares the sequencer to accept custom configurations by
    ensuring sequencer mode is off (this is a requirement to the enabling of
    sequencer configuration mode), disabling automatic gain and exposure, and
    turning sequencer configuration mode on.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** CONFIGURING SEQUENCER ***\n')
    try:
        result = True

        # Ensure sequencer is off for configuration
        #
        #  *** NOTES ***
        #  In order to configure a new sequence, sequencer configuration mode
        #  needs to be turned on. To do this, sequencer mode must be disabled.
        #  However, simply disabling sequencer mode might throw an exception if
        #  the current sequence is an invalid configuration.
        #
        #  Thus, in order to ensure that sequencer mode is disabled, we first
        #  check whether the current sequence is valid. If it
        #  isn't, then we know that sequencer mode is off and we can move on;
        #  if it is, then we can manually disable sequencer mode.
        #
        #  Also note that sequencer configuration mode needs to be off in order
        #  to manually disable sequencer mode. It should be off by default, so
        #  the example skips checking this.
        #
        #  Validate sequencer configuration
        node_sequencer_configuration_valid = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationValid'))
        if not PySpin.IsAvailable(node_sequencer_configuration_valid) \
                or not PySpin.IsReadable(node_sequencer_configuration_valid):
            print_retrieve_node_failure('node', 'SequencerConfigurationValid')
            return False

        sequencer_configuration_valid_yes = node_sequencer_configuration_valid.GetEntryByName('Yes')
        if not PySpin.IsAvailable(sequencer_configuration_valid_yes) \
                or not PySpin.IsReadable(sequencer_configuration_valid_yes):
            print_retrieve_node_failure('entry', 'SequencerConfigurationValid Yes')
            return False

        # If valid, disable sequencer mode; otherwise, do nothing
        node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
        if node_sequencer_configuration_valid.GetCurrentEntry().GetValue() == \
                sequencer_configuration_valid_yes.GetValue():
            if not PySpin.IsAvailable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
                print_retrieve_node_failure('node', 'SequencerMode')
                return False

            sequencer_mode_off = node_sequencer_mode.GetEntryByName('Off')
            if not PySpin.IsAvailable(sequencer_mode_off) or not PySpin.IsReadable(sequencer_mode_off):
                print_retrieve_node_failure('entry', 'SequencerMode Off')
                return False

            node_sequencer_mode.SetIntValue(sequencer_mode_off.GetValue())

        print('Sequencer mode disabled...')

        # Turn off automatic exposure
        #
        #  *** NOTES ***
        #  Automatic exposure prevents the manual configuration of exposure
        #  times and needs to be turned off for this example.
        #
        #  *** LATER ***
        #  Automatic exposure is turned back on at the end of the example in
        #  order to restore the camera to its default state.
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
            print_retrieve_node_failure('node', 'ExposureAuto')
            return False

        exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(exposure_auto_off) or not PySpin.IsReadable(exposure_auto_off):
            print_retrieve_node_failure('entry', 'ExposureAuto Off')
            return False

        node_exposure_auto.SetIntValue(exposure_auto_off.GetValue())

        print('Automatic exposure disabled...')

        # Turn off automatic gain
        #
        #  *** NOTES ***
        #  Automatic gain prevents the manual configuration of gain and needs
        #  to be turned off for this example.
        #
        #  *** LATER ***
        #  Automatic gain is turned back on at the end of the example in
        #  order to restore the camera to its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsAvailable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
            print_retrieve_node_failure('node', 'GainAuto')
            return False

        gain_auto_off = node_gain_auto.GetEntryByName('Off')
        if not PySpin.IsAvailable(gain_auto_off) or not PySpin.IsReadable(gain_auto_off):
            print_retrieve_node_failure('entry', 'GainAuto Off')
            return False

        node_gain_auto.SetIntValue(gain_auto_off.GetValue())

        print('Automatic gain disabled...')

        # Turn configuration mode on
        #
        # *** NOTES ***
        # Once sequencer mode is off, enabling sequencer configuration mode
        # allows for the setting of each state.
        #
        # *** LATER ***
        # Before sequencer mode is turned back on, sequencer configuration
        # mode must be turned back off.
        node_sequencer_configuration_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationMode'))
        if not PySpin.IsAvailable(node_sequencer_configuration_mode) \
                or not PySpin.IsWritable(node_sequencer_configuration_mode):
            print_retrieve_node_failure('node', 'SequencerConfigurationMode')
            return False

        sequencer_configuration_mode_on = node_sequencer_configuration_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(sequencer_configuration_mode_on)\
                or not PySpin.IsReadable(sequencer_configuration_mode_on):
            print_retrieve_node_failure('entry', 'SequencerConfigurationMode On')
            return False

        node_sequencer_configuration_mode.SetIntValue(sequencer_configuration_mode_on.GetValue())

        print('Sequencer configuration mode enabled...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result


def set_single_state(nodemap, sequence_number, width_to_set, height_to_set, exposure_time_to_set, gain_to_set):
    """
    This function sets a single state. It sets the sequence number, applies
    custom settings, selects the trigger type and next state number, and saves
    the state. The custom values that are applied are all calculated in the
    function that calls this one, run_single_camera().

    :param nodemap: Device nodemap.
    :param sequence_number: Sequence number.
    :param width_to_set: Width to set for sequencer.
    :param height_to_set: Height to set fpr sequencer.
    :param exposure_time_to_set: Exposure time to set for sequencer.
    :param gain_to_set: Gain to set for sequencer.
    :type nodemap: INodeMap
    :type sequence_number: int
    :type width_to_set: int
    :type height_to_set: int
    :type exposure_time_to_set: float
    :type gain_to_set: float
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Select the current sequence number
        #
        # *** NOTES ***
        # Select the index of the state to be set.
        #
        # *** LATER ***
        # The next state - i.e. the state to be linked to -
        # also needs to be set before saving the current state.
        node_sequencer_set_selector = PySpin.CIntegerPtr(nodemap.GetNode('SequencerSetSelector'))
        if not PySpin.IsAvailable(node_sequencer_set_selector) or not PySpin.IsWritable(node_sequencer_set_selector):
            print_retrieve_node_failure('node', 'SequencerSetSelector')
            return False

        node_sequencer_set_selector.SetValue(sequence_number)

        print('Setting state {}...'.format(sequence_number))

        # Set desired settings for the current state
        #
        # *** NOTES ***
        # Width, height, exposure time, and gain are set in this example. If
        # the sequencer isn't working properly, it may be important to ensure
        # that each feature is enabled on the sequencer. Features are enabled
        # by default, so this is not explored in this example.
        #
        # Changing the height and width for the sequencer is not available
        # for all camera models.
        #
        # Set width; width recorded in pixels
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
            width_inc = node_width.GetInc()

            if width_to_set % width_inc != 0:
                width_to_set = int(width_to_set / width_inc) * width_inc

            node_width.SetValue(width_to_set)

            print('\tWidth set to {}...'.format(node_width.GetValue()))

        else:
            print('\tUnable to set width; width for sequencer not available on all camera models...')

        # Set height; height recorded in pixels
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
            height_inc = node_height.GetInc()

            if height_to_set % height_inc != 0:
                height_to_set = int(height_to_set / height_inc) * height_inc

            node_height.SetValue(height_to_set)

            print('\tHeight set to %d...' % node_height.GetValue())

        else:
            print('\tUnable to set height; height for sequencer not available on all camera models...')

        # Set exposure time; exposure time recorded in microseconds
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            print_retrieve_node_failure('node', 'ExposureTime')
            return False

        exposure_time_max = node_exposure_time.GetMax()

        if exposure_time_to_set > exposure_time_max:
            exposure_time_to_set = exposure_time_max

        node_exposure_time.SetValue(exposure_time_to_set)

        print('\tExposure set to {0:.0f}...'.format(node_exposure_time.GetValue()))

        # Set gain; gain recorded in decibels
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsAvailable(node_gain) or not PySpin.IsWritable(node_gain):
            print_retrieve_node_failure('node', 'Gain')
            return False

        gain_max = node_gain.GetMax()

        if gain_to_set > gain_max:
            gain_to_set = gain_max

        node_gain.SetValue(gain_to_set)

        print('\tGain set to {0:.5f}...'.format(node_gain.GetValue()))

        # Set the trigger type for the current state
        #
        # *** NOTES ***
        # It is a requirement of every state to have its trigger source set.
        # The trigger source refers to the moment when the sequencer changes
        # from one state to the next.
        node_sequencer_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerTriggerSource'))
        if not PySpin.IsAvailable(node_sequencer_trigger_source) or not PySpin.IsWritable(node_sequencer_trigger_source):
            print_retrieve_node_failure('node', 'SequencerTriggerSource')
            return False

        sequencer_trigger_source_frame_start = node_sequencer_trigger_source.GetEntryByName('FrameStart')
        if not PySpin.IsAvailable(sequencer_trigger_source_frame_start) or \
                not PySpin.IsReadable(sequencer_trigger_source_frame_start):
            print_retrieve_node_failure('entry', 'SequencerTriggerSource FrameStart')
            return False

        node_sequencer_trigger_source.SetIntValue(sequencer_trigger_source_frame_start.GetValue())

        print('\tTrigger source set to start of frame...')

        # Set the next state in the sequence
        #
        # *** NOTES ***
        # When setting the next state in the sequence, ensure it does not
        # exceed the maximum and that the states loop appropriately.
        final_sequence_index = 4

        node_sequencer_set_next = PySpin.CIntegerPtr(nodemap.GetNode('SequencerSetNext'))
        if not PySpin.IsAvailable(node_sequencer_set_next) or not PySpin.IsWritable(node_sequencer_set_next):
            print('Unable to select next state. Aborting...\n')
            return False

        if sequence_number == final_sequence_index:
            node_sequencer_set_next.SetValue(0)
        else:
            node_sequencer_set_next.SetValue(sequence_number + 1)

        print('\tNext state set to {}...'.format(node_sequencer_set_next.GetValue()))

        # Save current state
        #
        # *** NOTES ***
        # Once all appropriate settings have been configured, make sure to
        # save the state to the sequence. Notice that these settings will be
        # lost when the camera is power-cycled.
        node_sequencer_set_save = PySpin.CCommandPtr(nodemap.GetNode('SequencerSetSave'))
        if not PySpin.IsAvailable(node_sequencer_set_save) or not PySpin.IsWritable(node_sequencer_set_save):
            print('Unable to save state. Aborting...\n')
            return False

        node_sequencer_set_save.Execute()

        print('Current state saved...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result


def configure_sequencer_part_two(nodemap):
    """"
    Now that the states have all been set, this function readies the camera
    to use the sequencer during image acquisition.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Turn configuration mode off
        #
        # *** NOTES ***
        # Once all desired states have been set, turn sequencer
        # configuration mode off in order to turn sequencer mode on.
        node_sequencer_configuration_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationMode'))
        if not PySpin.IsAvailable(node_sequencer_configuration_mode) \
                or not PySpin.IsWritable(node_sequencer_configuration_mode):
            print_retrieve_node_failure('node', 'SequencerConfigurationMode')
            return False

        sequencer_configuration_mode_off = node_sequencer_configuration_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(sequencer_configuration_mode_off)\
                or not PySpin.IsReadable(sequencer_configuration_mode_off):
            print_retrieve_node_failure('entry', 'SequencerConfigurationMode Off')
            return False

        node_sequencer_configuration_mode.SetIntValue(sequencer_configuration_mode_off.GetValue())

        print('Sequencer configuration mode disabled...')

        # Turn sequencer mode on
        #
        # *** NOTES ***
        # After sequencer mode has been turned on, the camera will begin using the
        # saved states in the order that they were set.
        #
        # *** LATER ***
        # Once all images have been captured, disable the sequencer in order
        # to restore the camera to its initial state.
        node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
        if not PySpin.IsAvailable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
            print_retrieve_node_failure('node', 'SequencerMode')
            return False

        sequencer_mode_on = node_sequencer_mode.GetEntryByName('On')
        if not PySpin.IsAvailable(sequencer_mode_on) or not PySpin.IsReadable(sequencer_mode_on):
            print_retrieve_node_failure('entry', 'SequencerMode On')
            return False

        node_sequencer_mode.SetIntValue(sequencer_mode_on.GetValue())

        print('Sequencer mode enabled...')

        # Validate sequencer settings
        #
        # *** NOTES ***
        # Once all states have been set, it is a good idea to
        # validate them. Although this node cannot ensure that the states
        # have been set up correctly, it does ensure that the states have
        # been set up in such a way that the camera can function.
        node_sequencer_configuration_valid = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationValid'))
        if not PySpin.IsAvailable(node_sequencer_configuration_valid) \
                or not PySpin.IsReadable(node_sequencer_configuration_valid):
            print_retrieve_node_failure('node', 'SequencerConfigurationValid')
            return False

        sequencer_configuration_valid_yes = node_sequencer_configuration_valid.GetEntryByName('Yes')
        if not PySpin.IsAvailable(sequencer_configuration_valid_yes) \
                or not PySpin.IsReadable(sequencer_configuration_valid_yes):
            print_retrieve_node_failure('entry', 'SequencerConfigurationValid Yes')
            return False

        if node_sequencer_configuration_valid.GetCurrentEntry().GetValue() != \
                sequencer_configuration_valid_yes.GetValue():
            print('Sequencer configuration not valid. Aborting...\n')
            return False

        print('Sequencer configuration valid...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result


def reset_sequencer(nodemap):
    """"
    This function restores the camera to its default state by turning sequencer mode
    off and re-enabling automatic exposure and gain.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Turn sequencer mode back off
        #
        # *** NOTES ***
        # The sequencer is turned off in order to return the camera to its default state.
        node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
        if not PySpin.IsAvailable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
            print_retrieve_node_failure('node', 'SequencerMode')
            return False

        sequencer_mode_off = node_sequencer_mode.GetEntryByName('Off')
        if not PySpin.IsAvailable(sequencer_mode_off) or not PySpin.IsReadable(sequencer_mode_off):
            print_retrieve_node_failure('entry', 'SequencerMode Off')
            return False

        node_sequencer_mode.SetIntValue(sequencer_mode_off.GetValue())

        print('Turning off sequencer mode...')

        # Turn automatic exposure back on
        #
        # *** NOTES ***
        # Automatic exposure is turned on in order to return the camera to its default state.
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsAvailable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
            exposure_auto_continuous = node_exposure_auto.GetEntryByName('Continuous')
            if PySpin.IsAvailable(exposure_auto_continuous) and PySpin.IsReadable(exposure_auto_continuous):
                node_exposure_auto.SetIntValue(exposure_auto_continuous.GetValue())
                print('Turning automatic exposure back on...')

        # Turn automatic gain back on
        #
        # *** NOTES ***
        # Automatic gain is turned on in order to return the camera to its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsAvailable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
            gain_auto_continuous = node_exposure_auto.GetEntryByName('Continuous')
            if PySpin.IsAvailable(gain_auto_continuous) and PySpin.IsReadable(gain_auto_continuous):
                node_gain_auto.SetIntValue(gain_auto_continuous.GetValue())
                print('Turning automatic gain mode back on...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
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


def acquire_images(cam, nodemap, nodemap_tldevice, timeout):
    """
    This function acquires and saves 10 images from a device.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param timeout: Timeout for image acquisition.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type timeout: int
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # Set acquisition mode to continuous
        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or \
                not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or \
                not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
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

                # Retrieve next received image and ensure image completion
                image_result = cam.GetNextImage(timeout)

                if image_result.IsIncomplete():
                    print('Image incomplete with image status {}...'.format(image_result.GetImageStatus()))

                else:

                    #  Print image information
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed image {}, width = {}, height = {}'.format(i, width, height))

                    #  Convert image to mono 8
                    image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)

                    # Create a unique filename
                    if device_serial_number:
                        filename = 'Sequencer-{}-{}.jpg'.format(device_serial_number, i)
                    else:  # if serial number is empty
                        filename = 'Sequencer-{}.jpg'.format(i)

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


def run_single_camera(cam):
    """
    This function acts very similarly to the run_single_camera() functions of other
    examples, except that the values for the sequences are also calculated here;
    please see NodeMapInfo example for more in-depth comments on setting up cameras.

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

        # Configure sequencer to be ready to set sequences
        result &= configure_sequencer_part_one(nodemap)
        if not result:
            return result

        # Set sequences
        #
        # *** NOTES ***
        # In the following section, the sequencer values are calculated. This
        # section does not appear in the configuration, as the values
        # calculated are somewhat arbitrary: width and height are both set to
        # 25% of their maximum values, incrementing by 10%; exposure time is
        # set to its minimum, also incrementing by 10% of its maximum; and gain
        # is set to its minimum, incrementing by 2% of its maximum.
        num_sequences = 5

        # Retrieve maximum width; width recorded in pixels
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if not PySpin.IsAvailable(node_width) or not PySpin.IsReadable(node_width):
            print('Unable to retrieve maximum width. Aborting...\n')
            return False

        width_max = node_width.GetMax()

        # Retrieve maximum height; height recorded in pixels
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if not PySpin.IsAvailable(node_height) or not PySpin.IsReadable(node_height):
            print('Unable to retrieve maximum height. Aborting...\n')
            return False

        height_max = node_height.GetMax()

        # Retrieve maximum exposure time; exposure time recorded in microseconds
        exposure_time_max_to_set = 2000000

        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsReadable(node_exposure_time):
            print('Unable to retrieve maximum exposure time. Aborting...\n')
            return False

        exposure_time_max = node_exposure_time.GetMax()

        if exposure_time_max > exposure_time_max_to_set:
            exposure_time_max = exposure_time_max_to_set

        # Retrieve maximum gain; gain recorded in decibels
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsAvailable(node_exposure_time) or not PySpin.IsReadable(node_exposure_time):
            print('Unable to retrieve maximum gain. Aborting...\n')
            return False

        gain_max = node_gain.GetMax()

        # Set initial values
        width_to_set = width_max / 4
        height_to_set = height_max / 4
        exposure_time_to_set = node_exposure_time.GetMin()
        gain_to_set = node_gain.GetMin()

        # Set custom values of each sequence
        for sequence_num in range(num_sequences):
            result &= set_single_state(nodemap,
                                       sequence_num,
                                       int(width_to_set),
                                       int(height_to_set),
                                       exposure_time_to_set,
                                       gain_to_set)
            if not result:
                return result

            # Increment values
            width_to_set += width_max / 10
            height_to_set += height_max / 10
            exposure_time_to_set += exposure_time_max / 10.0
            gain_to_set += gain_max / 50.0

        # Calculate appropriate acquisition grab timeout window based on exposure time
        # Note: exposure_time_to_set is in microseconds and needs to be converted to milliseconds
        timeout = (exposure_time_to_set / 1000) + 1000

        # Configure sequencer to acquire images
        result &= configure_sequencer_part_two(nodemap)
        if not result:
            return result

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice, int(timeout))

        # Reset sequencer
        result &= reset_sequencer(nodemap)

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
    print('Library version: {}.{}.{}.{}\n'.format(version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: {}\n'.format(num_cameras))

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

        print('Running example for camera {}...\n'.format(i))

        result &= run_single_camera(cam)
        print('Camera {} example complete...\n'.format(i))

    # Release reference to camera
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
