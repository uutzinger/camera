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
# NodeMapCallback.py shows how to use nodemap callbacks. It relies
# on information provided in the Enumeration, Acquisition, and NodeMapInfo
# examples. As callbacks are very similar to events, it may be a good idea to
# explore this example prior to tackling the events examples.
#
# This example focuses on creating, registering, using, and unregistering
# callbacks. A callback requires a callback class with a callback function signature,
# which allows it to be registered to and access a node. Events follow this same pattern.
#
# Once comfortable with NodeMapCallback, we suggest checking out any of the
# events examples: EnumerationEvents, ImageEvents, or Logging.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import PySpin
import sys


NUM_IMAGES = 10  # number of images to grab

class HeightNodeCallback(PySpin.NodeCallback):
    """
    This is the first of three callback classes.  This callback will be registered to the height node.
    Node callbacks must inherit from NodeCallback, and must implement CallbackFunction with the same function signature.

    NOTE: Instances of callback classes must not go out of scope until they are deregistered, otherwise segfaults
    will occur.
    """
    def __init__(self):
        super(HeightNodeCallback, self).__init__()

    def CallbackFunction(self, node):
        """
        This function gets called when the height node changes and triggers a callback.

        :param node: Height node.
        :type node: INode
        :rtype: None
        """
        node_height = PySpin.CIntegerPtr(node)
        print('Height callback message:\n\tLook! Height changed to %f...\n' % node_height.GetValue())


class GainNodeCallback(PySpin.NodeCallback):
    """
    This is the second callback class, registered to the gain node.
    """
    def __init__(self):
        super(GainNodeCallback, self).__init__()

    def CallbackFunction(self, node):
        """
        This function gets called when the gain node changes and triggers a callback.

        :param node: Gain node.
        :type node: INode
        :rtype: None
        """
        node_gain = PySpin.CFloatPtr(node)
        print('Gain callback message:\n\tLook! Gain changed to %f...\n' % node_gain.GetValue())


class EventNodeCallback(PySpin.NodeCallback):
    """
    This is the third callback class, registered to event feature nodes.
    """
    def __init__(self):
        super(EventNodeCallback, self).__init__()

    def CallbackFunction(self, event_node):
        """
        This function gets called when the event node changes and triggers a callback.

        :param node: Event node.
        :type node: INode
        :rtype: None
        """
        node_name = event_node.GetName()

        if event_node.GetPrincipalInterfaceType() == PySpin.intfIInteger:
            integer = PySpin.CIntegerPtr(event_node)
            print('%s callback message:\n\tValue changed to %f...\n' % (node_name, integer.GetValue()))
        elif event_node.GetPrincipalInterfaceType() == PySpin.intfIBoolean:
            boolean = PySpin.CBooleanPtr(event_node)
            print('%s callback message:\n\tValue changed to %f...\n' % (node_name, boolean.GetValue()))
        elif event_node.GetPrincipalInterfaceType() == PySpin.intfIFloat:
            float = PySpin.CFloatPtr(event_node)
            print('%s callback message:\n\tValue changed to %f...\n' % (node_name, float.GetValue()))
        elif event_node.GetPrincipalInterfaceType() == PySpin.intfIString:
            string = PySpin.CStringPtr(event_node)
            print('%s callback message:\n\tValue changed to %s...\n' % (node_name, string.GetValue()))
        else:
            print('%s callback message:\n\t node with type %s was updated.\n' % (node_name, event_node.GetPrincipalInterfaceType()))


def configure_callbacks(nodemap, callback_list):
    """
    This function sets up the example by disabling automatic gain, creating the callbacks, and registering them to
    their specific nodes.

    :param nodemap: Device nodemap.
    :param callback_list: List of NodeCallbacks
    :type nodemap: INodeMap
    :returns: tuple (result, callback_list)
        WHERE
        result is True if successful, False otherwise
        callback_list is the list of NodeCallback instances registered 
        callback_height is the HeightNodeCallback instance registered to the height node
        callback_gain is the GainNodeCallback instance registered to the gain node
    :rtype: (bool, list)
    """
    print('\n*** CONFIGURING CALLBACKS ***\n')
    try:
        result = True

        # Turn off automatic gain
        #
        # *** NOTES ***
        # Automatic gain prevents the manual configuration of gain and needs to
        # be turned off for this example.
        #
        # *** LATER ***
        # Automatic exposure is turned off at the end of the example in order
        # to restore the camera to its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsReadable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
            print('Unable to disable automatic gain (node retrieval). Aborting...')
            return False, None

        node_gain_auto_off = PySpin.CEnumEntryPtr(node_gain_auto.GetEntryByName('Off'))
        if not PySpin.IsReadable(node_gain_auto_off):
            print('Unable to disable automatic gain (enum entry retrieval). Aborting...')
            return False, None

        node_gain_auto.SetIntValue(node_gain_auto_off.GetValue())
        print('Automatic gain disabled...')

        # Register callback to height node
        #
        # *** NOTES ***
        # Callbacks need to be registered to nodes, which should be writable
        # if the callback is to ever be triggered. Also ensure that the callback
        # instance does not go out of scope, as it will get garbage-collected
        # and a segfault will result once the callback actually occurs.
        #
        # *** LATER ***
        # Each callback needs to be unregistered individually before releasing
        # the system or an exception will be thrown.
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if not PySpin.IsReadable(node_height) or not PySpin.IsWritable(node_height):
            print('Unable to retrieve height. Aborting...\n')
            return False, None

        print('Height ready...')

        callback_height = HeightNodeCallback()
        PySpin.RegisterNodeCallback(node_height.GetNode(), callback_height)
        callback_list.append(callback_height)

        print('Height callback registered...')

        # Register callback to gain node
        #
        # *** NOTES ***
        # Depending on the specific goal of the function, it can be important
        # to notice the node type that a callback is registered to. Notice in
        # the callback functions above that the callback registered to height
        # casts its node as an integer whereas the callback registered to gain
        # casts as a float.
        #
        # *** LATER ***
        # Each callback needs to be unregistered individually before releasing
        # the system or an exception will be thrown.
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsReadable(node_gain) or not PySpin.IsWritable(node_gain):
            print('Unable to retrieve gain. Aborting...\n')
            return False, None

        print('Gain ready...')

        callback_gain = GainNodeCallback()
        PySpin.RegisterNodeCallback(node_gain.GetNode(), callback_gain)
        callback_list.append(callback_gain)

        print('Gain callback registered...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False, None

    return result, callback_list


def configure_event_callbacks(nodemap, callback_list):

    print('\n*** CONFIGURING EVENT CALLBACKS ***\n')
    try:
        result = True

        #
        # Retrieve event selector
        #
        # *** NOTES ***
        # Each type of event must be enabled individually. This is done
        # by retrieving "EventSelector" (an enumeration node) and then enabling
        # the specific event on "EventNotification" (another enumeration node).
        #
        event_selector = PySpin.CEnumerationPtr(nodemap.GetNode("EventSelector"))
        if not PySpin.IsReadable(event_selector) or not PySpin.IsWritable(event_selector):
            print('Unable to retrieve event selector entries. Skipping...\n')
            return None, callback_list

        entries = [PySpin.CEnumEntryPtr(event_selector_entry) for event_selector_entry in event_selector.GetEntries()]

        print('Enabling event selector entries...')

        #
        # Enable device events
        #
        # *** NOTES ***
        # In order to enable a specific event, the event selector and event
        # notification nodes (both of type enumeration) must work in unison.
        # The desired event must first be selected on the event selector node
        # and then enabled on the event notification node.
        #
        for event_selector_entry in entries:
            # Select entry on selector node
            if not PySpin.IsReadable(event_selector_entry):
                # Skip if node fails
                continue

            event_selector.SetIntValue(event_selector_entry.GetValue())

            # Retrieve event notification node (an enumeration node)
            event_notification = PySpin.CEnumerationPtr(nodemap.GetNode("EventNotification"))

            # Retrieve entry node to enable device event
            if not PySpin.IsReadable(event_notification):
                # Skip if node fails
                continue

            event_notification_on = PySpin.CEnumEntryPtr(event_notification.GetEntryByName("On"))

            if not PySpin.IsReadable(event_notification_on):
                # Skip if node fails
                continue

            if not PySpin.IsWritable(event_notification):
                # Skip if node fails
                continue

            event_notification.SetIntValue(event_notification_on.GetValue())

            print('\n\t%s: enabled\n' % event_selector_entry.GetDisplayName())

            # Register Event Data callbacks
            entry_name = event_selector_entry.GetSymbolic()
            data_category_name = 'Event' + entry_name + 'Data'
            data_category = PySpin.CCategoryPtr(nodemap.GetNode(data_category_name))

            if data_category:
                features = data_category.GetFeatures()

                for feature in features:
                    #
                    # Register callback to event data node
                    #
                    # *** LATER ***
                    # Each callback needs to be unregistered individually before releasing
                    # the system or an exception will be thrown.
                    #
                    feature_node = PySpin.CNodePtr(feature)
                    feature_name = feature_node.GetDisplayName()

                    callback_feature = EventNodeCallback()
                    PySpin.RegisterNodeCallback(feature.GetNode(), callback_feature)
                    callback_list.append(callback_feature)

                    print('\t%s callback registered...' %feature_node.GetName())

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False, None

    return result, callback_list


def reset_events(nodemap):

    try:
        result = True

        # Retrieve event selector
        event_selector = PySpin.CEnumerationPtr(nodemap.GetNode("EventSelector"))
        if not PySpin.IsReadable(event_selector) or not PySpin.IsWritable(event_selector):
            print('Unable to retrieve event selector entries. Skipping...\n')
            return result

        entries = [PySpin.CEnumEntryPtr(event_selector_entry) for event_selector_entry in event_selector.GetEntries()]

        print('Disabling event selector entries...')

        # Disable device events
        for event_selector_entry in entries:
            # Select entry on selector node
            if not PySpin.IsReadable(event_selector_entry):
                # Skip if node fails
                continue

            event_selector.SetIntValue(event_selector_entry.GetValue())

            # Retrieve event notification node (an enumeration node)
            event_notification = PySpin.CEnumerationPtr(nodemap.GetNode("EventNotification"))

            # Retrieve entry node to enable device event
            if not PySpin.IsReadable(event_notification):
                # Skip if node fails
                continue

            event_notification_off = PySpin.CEnumEntryPtr(event_notification.GetEntryByName("Off"))

            if not PySpin.IsReadable(event_notification_off):
                # Skip if node fails
                continue

            if not PySpin.IsWritable(event_notification):
                # Skip if node fails
                continue

            event_notification.SetIntValue(event_notification_off.GetValue())

            print('\n\t%s: disabled\n' % event_selector_entry.GetDisplayName())

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False, None

    return result


def change_height_and_gain(nodemap):
    """
    This function demonstrates the triggering of the nodemap callbacks. First it
    changes height, which executes the callback registered to the height node, and
    then it changes gain, which executes the callback registered to the gain node.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """        # if not PySpin.IsReadable(event_node):
        #     print('Event callback triggered but node is not readable...\n')
        #     return
    print('\n***CHANGE HEIGHT & GAIN ***\n')

    try:
        result = True

        # Change height to trigger height callback
        # 
        # *** NOTES ***
        # Notice that changing the height only triggers the callback function
        # registered to the height node.
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if not PySpin.IsReadable(node_height) or not PySpin.IsWritable(node_height) \
                or node_height.GetInc() == 0 or node_height.GetMax() == 0:
            print('Unable to retrieve height. Aborting...')
            return False

        height_to_set = node_height.GetMax()

        print('Regular function message:\n\tHeight about to be changed to %i...\n' % height_to_set)

        node_height.SetValue(height_to_set)

        # Change gain to trigger gain callback
        #
        # *** NOTES ***
        # The same is true of changing the gain node; changing a node will
        # only ever trigger the callback function (or functions) currently
        # registered to it.
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsReadable(node_gain) or not PySpin.IsWritable(node_gain) or node_gain.GetMax() == 0:
            print('Unable to retrieve gain...')
            return False

        gain_to_set = node_gain.GetMax() / 2.0

        print('Regular function message:\n\tGain about to be changed to %f...\n' % gain_to_set)
        node_gain.SetValue(gain_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def reset_callbacks(nodemap, callback_list):
    """
    This function cleans up the example by deregistering the callbacks and 
    turning automatic gain back on.

    :param nodemap: Device nodemap.
    :param callback_list: List of node callback instances to deregister.
    :type nodemap: INodeMap
    :type callback_list: List
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        for callback in callback_list:
            # Deregister callbacks
            #
            # *** NOTES ***
            # It is important to deregister each callback function from each node
            # that it is registered to.
            PySpin.DeregisterNodeCallback(callback)

        print('Callbacks deregistered...')

        # Turn automatic gain back on
        # 
        # *** NOTES ***
        # Automatic gain is turned back on in order to restore the camera to 
        # its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsReadable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
            print('Unable to enable automatic gain (node retrieval). Aborting...')
            return False

        node_gain_auto_continuous = PySpin.CEnumEntryPtr(node_gain_auto.GetEntryByName('Continuous'))
        if not PySpin.IsReadable(node_gain_auto_continuous):
            print('Unable to enable automatic gain (enum entry retrieval). Aborting...')
            return False

        node_gain_auto.SetIntValue(node_gain_auto_continuous.GetValue())
        print('Automatic gain disabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    This function acquires and saves 10 images from a device.

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

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        #  Retrieve device serial number for filename
        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Retrieve, convert, and save images

        # Create ImageProcessor instance for post processing images
        processor = PySpin.ImageProcessor()

        # Set default image processor color processing method
        #
        # *** NOTES ***
        # By default, if no specific color processing algorithm is set, the image
        # processor will default to NEAREST_NEIGHBOR method.
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        for i in range(NUM_IMAGES):
            try:

                #  Retrieve next received image
                image_result = cam.GetNextImage(1000)

                #  Ensure image completion
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                else:

                    #  Print image information; height and width recorded in pixels
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                    #  Convert image to mono 8
                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)

                    #  Release image
                    image_result.Release()
                    print('')

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        #  End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

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

    :param cam: Camera to setup and run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Retrieve TL stream nodemap
        nodemap_tlstream = cam.GetTLStreamNodeMap()

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Configure callbacks
        callback_list = []
        err, callback_list = configure_callbacks(nodemap, callback_list)
        if not err:
            return err
        
        # Configure event callbacks on remote device
        err, callback_list = configure_event_callbacks(nodemap, callback_list)
        if err is False and err is not None:
            return err
        
        # Configure event callbacks on local device
        err, callback_list = configure_event_callbacks(nodemap_tldevice, callback_list)
        if err is False and err is not None:
            return err
        
        # Configure event callbacks on local stream
        err, callback_list = configure_event_callbacks(nodemap_tlstream, callback_list)
        if err is False and err is not None:
            return err

        # Change height and gain to trigger callbacks
        result &= change_height_and_gain(nodemap)

        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Reset callbacks
        result &= reset_callbacks(nodemap, callback_list)

        # Reset event notifications on remote device
        result &= reset_events(nodemap)
        
        # Reset event notifications on local device
        result &= reset_events(nodemap_tldevice)

        # Reset event notifications on local stream
        result &= reset_events(nodemap_tlstream)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

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

    # Release instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
