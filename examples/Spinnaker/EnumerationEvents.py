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
#  	EnumerationEvents.py explores arrival and removal events on interfaces and the system.
#   It relies on information provided in the Enumeration, Acquisition, and NodeMapInfo examples.
#
#  	It can also be helpful to familiarize yourself with the NodeMapCallback example,
#   as nodemap callbacks follow the same general procedure as events, but with a few less steps.
#
#  	This example creates two user-defined classes: InterfaceEventHandler and SystemEventHandler.
#   These child classes allow the user to define properties, parameters, and the event handler itself
#   while the parent classes - DeviceArrivalEventHandler, DeviceRemovalEventHandler, and InterfaceEventHandler -
#   allow the child classes to interface with Spinnaker.

import PySpin


class SystemEventHandler(PySpin.InterfaceEventHandler):
    """
    This class defines the properties and methods for device arrivals and removals
    on the system. Take special note of the signatures of the OnDeviceArrival()
    and OnDeviceRemoval() methods. 
    All three event handler types - DeviceArrivalEventHandler, DeviceRemovalEventHandler, 
    and InterfaceEventHandler - can be registered to interfaces, the system, or both.
    However, in Spinnaker Python, an enumeration event handler must inherit from
    InterfaceEventHandler if we want to handle both arrival and removal events.

    *** NOTES ***
    Registering an interface event handler to the system is basically the same thing
    as registering that event handler to all interfaces, with the added benefit of 
    not having to manage newly arrived or newly removed interfaces. In order to manually
    manage newly arrived or removed interfaces, one would need to implement interface
    arrival/removal event handlers, which are not yet supported in the Spinnaker Python API.
    """
    def __init__(self, system):
        """
        Constructor. This sets the system instance.

        :param system: Instance of the system.
        :type system: SystemPtr
        :rtype: None
        """
        super(SystemEventHandler, self).__init__()
        self.system = system


    def OnDeviceArrival(self, serial_number):
        """
        This method defines the arrival event on the system. It prints out
        the device serial number of the camera arriving and the number of
        cameras currently connected. The argument is the serial number of 
        the camera that triggered the arrival event.

        :param serial_number: gcstring representing the serial number of the arriving camera.
        :type serial_number: gcstring
        :return: None
        """
        cam_list = self.system.GetCameras()
        count = cam_list.GetSize()
        print('System event handler:')
        print('\tDevice %i has arrived on the system.' % serial_number)
        print('\tThere %s %i %s on the system.' % ('is' if count == 1 else 'are',
                                                   count,
                                                   'device' if count == 1 else 'devices'))


    def OnDeviceRemoval(self, serial_number):
        """
        This method defines the removal event on the system. It prints out the
        device serial number of the camera being removed and the number of cameras
        currently connected. The argument is the serial number of the camera that 
        triggered the removal event.

        :param serial_number: gcstring representing the serial number of the removed camera.
        :type serial_number: gcstring
        :return: None
        """
        cam_list = self.system.GetCameras()
        count = cam_list.GetSize()
        print('System event handler:')
        print('\tDevice %i was removed from the system.' % serial_number)
        print('\tThere %s %i %s on the system.' % ('is' if count == 1 else 'are',
                                                   count,
                                                   'device' if count == 1 else 'devices'))


def check_gev_enabled(system):
    """
    This function checks if GEV enumeration is enabled on the system.

    :param system: Current system instance.
    :type system: SystemPtr

    """

    # Retrieve the System TL NodeMap and EnumerateGEVInterfaces node
    system_node_map = system.GetTLNodeMap()
    node_gev_enumeration = PySpin.CBooleanPtr(system_node_map.GetNode('EnumerateGEVInterfaces'))

    # Ensure the node is valid
    if not PySpin.IsAvailable(node_gev_enumeration) or not PySpin.IsReadable(node_gev_enumeration):
        print('EnumerateGEVInterfaces node is unavailable or unreadable. Aborting...')
        return

    # Check if node is enabled
    gev_enabled = node_gev_enumeration.GetValue()
    if not gev_enabled:
        print('\nWARNING: GEV Enumeration is disabled.')
        print('If you intend to use GigE cameras please run the EnableGEVInterfaces shortcut\n'
              'or set EnumerateGEVInterfaces to true and relaunch your application.\n')
        return
    print('GEV enumeration is enabled. Continuing..')


def main():
    """
    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :rtype: None
    """
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Check if GEV enumeration is enabled
    check_gev_enabled(system)

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    
    num_cams = cam_list.GetSize()
    
    print('Number of cameras detected: %i' % num_cams)

    # Retrieve list of interfaces from the system
    #
    # *** NOTES ***
    # MacOS interfaces are only registered if they are active. 
    # For this example to have the desired outcome all devices must be connected 
    # at the beginning and end of this example in order to register and deregister 
    # an event handler on each respective interface.
    iface_list = system.GetInterfaces()

    num_ifaces = iface_list.GetSize()

    print('Number of interfaces detected: %i' % num_ifaces)

    print('*** CONFIGURING ENUMERATION EVENTS *** \n')

    # Create interface event handler for the system
    #
    # *** NOTES ***
    # The SystemEventHandler has been constructed to accept a system object in
    # order to print the number of cameras on the system.
    system_event_handler = SystemEventHandler(system)

    # Register interface event handler for the system
    #
    # *** NOTES ***
    # Arrival, removal, and interface event handlers can all be registered to
    # interfaces or the system. Do not think that interface event handlers can only be
    # registered to an interface. An interface event handler is merely a combination
    # of an arrival and a removal event handler.
    #
    # *** LATER ***
    # Arrival, removal, and interface event handlers must all be unregistered manually.
    # This must be done prior to releasing the system and while they are still
    # in scope.
    system.RegisterInterfaceEventHandler(system_event_handler)

    # Wait for user to plug in and/or remove camera devices
    input('\nReady! Remove/Plug in cameras to test or press Enter to exit...\n')

    # Unregister system event handler from system object
    #
    # *** NOTES ***
    # It is important to unregister all arrival, removal, and interface event handlers
    # registered to the system.
    system.UnregisterInterfaceEventHandler(system_event_handler)

    # Delete system event handler, which has a system reference
    del system_event_handler
    print('Event handler unregistered from system...')

    # Clear camera list before releasing system
    cam_list.Clear()

    # Clear interface list before releasing system
    iface_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')

if __name__ == '__main__':
    main()
