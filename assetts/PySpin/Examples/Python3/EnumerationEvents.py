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
# EnumerationEvents.py explores arrival and removal events on interfaces and the system.
# It relies on information provided in the Enumeration, Acquisition, and NodeMapInfo examples.
#
# It can also be helpful to familiarize yourself with the NodeMapCallback example,
# as nodemap callbacks follow the same general procedure as events, but with a few less steps.
#
# This example creates two user-defined classes: InterfaceEventHandler and SystemEventHandler.
# These child classes allow the user to define properties, parameters, and the event handler itself
# while the parent classes - DeviceArrivalEventHandler, DeviceRemovalEventHandler, and InterfaceEventHandler -
# allow the child classes to interface with Spinnaker.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import PySpin
from PySpin import IsReadable

from threading import Lock


class InterfaceEventHandler(PySpin.InterfaceEventHandler):
    """
    This class defines the properties and methods for device arrivals and removals
    on an interface. Take special note of the signatures of the OnDeviceArrival()
    and OnDeviceRemoval() methods. In Spinnaker Python, an enumeration event handler must
    inherit from InterfaceEventHandler if we want to handle both arrival and removal events.
    """

    def __init__(self, system=None, interface=None, interface_id="", register_to_system=False):
        # *** NOTES ***
        # When constructing a generic InterfaceEventHandler to be registered to the system,
        # the handler will not have knowledge of which interface triggered the event callbacks.
        # On the other hand, InterfaceEventHandler does not need knowledge about the system if
        # we are constructing it to be registered to a specific interface.
        super(InterfaceEventHandler, self).__init__()

        self.system = system
        self.interface = interface
        self.interfaceId = interface_id
        self.register_to_system = register_to_system

    def print_generic_handler_message(self, device_count):
        """
        Helper function to print the number of devices on an interface event registered to the system.

        :param device_count: the number of devices connected to the system
        :return: None
        """
        print('Generic interface event handler:')
        print("\tThere {0} {1} {2} on the system.\n".format('is' if device_count == 1 else 'are',
              device_count, 'device' if device_count == 1 else 'devices'))

    def OnDeviceArrival(self, camera):
        """
        This method defines the arrival event on the system. It prints out the device serial
        number of the camera arriving and the number of cameras currently connected. The argument
        is the serial number of the camera that triggered the arrival event.

        :param camera: PySpin camera object of the arriving camera.
        :type camera: PySpin.Camera
        :return: None
        """
        if self.register_to_system:
            cam_list = self.system.GetCameras()
            count = cam_list.GetSize()
            self.print_generic_handler_message(count)
        else:
            device_serial_number = ''
            nodemap_tldevice = camera.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            print('Interface event handler:')
            print('\tDevice {0} has arrived on interface {1}\n'.format(device_serial_number, self.interfaceId))

    def OnDeviceRemoval(self, camera):
        """
        This method defines the removal event on the system. It prints out the device serial number
        of the camera being removed and the number of cameras currently connected. The argument is
        the serial number of the camera that triggered the removal event.

        :param camera: PySpin camera object of the removed camera.
        :type camera: PySpin.Camera
        :return: None
        """
        if self.register_to_system:
            cam_list = self.system.GetCameras()
            count = cam_list.GetSize()
            self.print_generic_handler_message(count)
        else:
            device_serial_number = ''
            nodemap_tldevice = camera.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()

            print('Interface event handler:')
            print('\tDevice {0} was removed from interface {1}.\n'.format(device_serial_number, self.interfaceId))

    def get_interface_id(self):
        return self.interfaceId


class SystemEventHandler(PySpin.SystemEventHandler):
    """
    This class defines the properties and methods of the system event handler that handles
    interface arrival and removal events on the system. Take special note of the signatures of
    the OnInterfaceArrival() and OnInterfaceRemoval() methods.
    Interface enumeration event handlers must inherit from PySpin.SystemEventHandler.
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
        self.interface_event_handler_on_system = None
        self.interface_event_handlers = {}
        self.mutex = Lock()

    def OnInterfaceArrival(self, iface):
        """
        This method defines the interface arrival event callback on the system.
        It first prints the ID of the arriving interface, then registers the interface
        event on the newly arrived interface.

        ***NOTES***
        Only arrival events for GEV interfaces are currently supported.

        :param interfaceID: the interface ID
        :type interfaceID: gcstring
        :return: None
        """
        nodemap_tlinterface = iface.GetTLNodeMap()
        interface_id_node = PySpin.CStringPtr(nodemap_tlinterface.GetNode("InterfaceID"))
        interface_id = interface_id_node.GetValue()

        print('System event handler:')
        print('\tInterface \'%s\' has arrived on the system.\n' % interface_id)

        # UpdateInterfaceList() only updates newly arrived or newly removed interfaces.
        # In particular, after this call:
        #
        # - Any pre-existing interfaces will still be valid
        # - Newly removed interfaces will be invalid.
        #
        # ***NOTES***
        # - Invalid interfaces will be re-validated if the interface comes back (arrives)
        # with the same interface ID. If the interface ID changes, you can use the pointer
        # populated by this callback or you must get the new interface object from the updated
        # interface list in order to access this interface.
        #
        # - Interface indices used to access an interface with GetInterfaces() may change after
        # updating the interface list. The interface at a particular index cannot be expected to
        # remain at that index after calling UpdateInterfaceList().
        self.system.UpdateInterfaceList()

        cam_list = iface.GetCameras()
        cam_num = cam_list.GetSize()
        for cam_idx in range(cam_num):
            cam = cam_list.GetByIndex(cam_idx)
            node_map_tl_device = cam.GetTLDeviceNodeMap()
            device_serial_num = node_map_tl_device.GetNode("DeviceSerialNumber")

            if IsReadable(device_serial_num):
                device_serial_num_value = PySpin.CValuePtr(device_serial_num).ToString()
                print('Device {0} is connected to interface {1}'.format(device_serial_num_value, interface_id))

        '''Create interface event'''
        self.mutex.acquire()
        try:
            interface_event_handler = InterfaceEventHandler(None, iface, interface_id, False)
            self.interface_event_handlers.update({interface_id: interface_event_handler})

            # Register interface event listener
            iface.RegisterEventHandler(interface_event_handler)
            print('Event handler registered to interface \'%s\' ...\n' % interface_id)
        except PySpin.SpinnakerException as ex:
            print('Error registering interface event handler to {0}: {1}'.format(interface_id, ex))
        finally:
            self.mutex.release()

    def OnInterfaceRemoval(self, iface):
        """
        This method defines the interface removal event callback on the system.
        It prints the ID of the interface removed.

        ***NOTES***
        Only removal events for GEV interfaces are currently supported.

        :param interfaceID: the interface ID of the host adapter
        :type interfaceID: gcstring
        :return: None
        """
        nodemap_tlinterface = iface.GetTLNodeMap()
        interface_id_node = PySpin.CStringPtr(nodemap_tlinterface.GetNode("InterfaceID"))
        interface_id = interface_id_node.GetValue()

        print('System event handler:')
        print('\tInterface \'%s\' was removed from the system.\n' % interface_id)

        # Interface indices used to access an interface with GetInterfaces() may change after updating
        # the interface list. The interface at a particular index cannot be expected to remain at that
        # index after calling UpdateInterfaceList()
        self.system.UpdateInterfaceList()

        # Find the vent handler that was registered to the removed interface and remove it.
        # Interface event handlers are automatically unregistered when the interface is removed so that it
        # is not necessary to manually unregister them.
        self.mutex.acquire()
        try:
            for ifaceId in self.interface_event_handlers.items():
                if ifaceId == interface_id:
                    self.interface_event_handlers.pop(ifaceId)
                    break
        except PySpin.SpinnakerException as ex:
            print('Error erasing event handler from interface {0} : {1}\n'.format(interface_id, ex))
        finally:
            self.mutex.release()

    def register_interface_event_to_system(self):
        if self.interface_event_handler_on_system is None:
            # Create interface event handler for the system
            #
            # ***NOTES***
            # The InterfaceEventHandler has been constructed to accept a system object in order
            # to print the number of cameras on the system.
            self.interface_event_handler_on_system = InterfaceEventHandler(self.system, None, "", True)

        # Register interface event handler for the system
        #
        # *** NOTES ***
        # Arrival, removal, and interface event handlers can all be registered to interfaces or the system.
        # Do not think that interface event handlers can only be registered to an interface. An interface event
        # is merely a combination of an arrival and a removal event.
        # Only arrival and removal events for GEV interfaces are currently supported.
        #
        # *** LATER ***
        # Arrival, removal, and interface event handlers must all be unregistered manually. This must be done
        # prior to releasing the system and while they are still in scope.
        self.system.RegisterEventHandler(self.interface_event_handler_on_system)
        print('Interface event handler registered on the system...')

    def unregister_interface_event_to_system(self):
        # Unregister interface vent handler from system object
        #
        # *** NOTES ***
        # It is important to unregister all arrival, removal, and interface event handlers registered to the system.
        if self.interface_event_handler_on_system is not None:
            self.system.UnregisterEventHandler(self.interface_event_handler_on_system)
            print('Interface event handler unregistered from system...')
            self.interface_event_handler_on_system = None

    def register_all_interface_events(self):
        self.mutex.acquire()
        if not self.interface_event_handlers:
            self.interface_event_handlers.clear()
        self.mutex.release()

        interface_list = self.system.GetInterfaces()
        interface_num = interface_list.GetSize()

        # Create and register interface event handler to each interface
        #
        # *** NOTES ***
        # The process of event handler creation and registration on interfaces is similar to the process of
        # event handler creation and registration on the system. The InterfaceEventHandler class has been
        # written to accept an interface and an interface ID to differentiate between the interfaces.
        #
        # *** LATER ***
        # Arrival, removal, and interface event handlers must all be unregistered manually. This must be done
        # prior to releasing the system and while they are still in scope.
        for ifaceIdx in range(interface_num):
            interface = interface_list.GetByIndex(ifaceIdx)
            node_map = interface.GetTLNodeMap()

            interface_id_node = node_map.GetNode("InterfaceID")
            if not IsReadable(interface_id_node):
                continue

            interface_id = PySpin.CStringPtr(interface_id_node).ToString()
            self.mutex.acquire()
            try:
                interface_event_handler = InterfaceEventHandler(None, interface, interface_id, False)
                self.interface_event_handlers.update({interface_id: interface_event_handler})

                # Register interface event handler
                interface.RegisterEventHandler(interface_event_handler)
                print('Event handler registered to interface \'%s\' ...' % interface_id)
            except PySpin.SpinnakerException as ex:
                print('Error registering event handler to interface {0} : {1}\n'.format(interface_id, ex))
            finally:
                self.mutex.release()
        print('')

    def unregister_all_interface_events(self):
        interface_list = self.system.GetInterfaces(False)
        interface_num = interface_list.GetSize()

        self.mutex.acquire()
        # Unregister interface event handler from each interface
        #
        # *** NOTES ***
        # It is important to unregister all arrival, removal, and interface event handlers from all
        # interfaces that they may be registered to.
        for ifaceIdx in range(interface_num):
            # Select interface
            interface = interface_list.GetByIndex(ifaceIdx)
            node_map = interface.GetTLNodeMap()

            interface_id_node = node_map.GetNode("InterfaceID")
            # Ensure the node is valid
            if not IsReadable(interface_id_node):
                continue

            interface_id = PySpin.CStringPtr(interface_id_node).ToString()

            try:
                for ifaceId, iface_event_handler in self.interface_event_handlers.items():
                    if interface_id == ifaceId:
                        interface.UnregisterEventHandler(iface_event_handler)
            except PySpin.SpinnakerException as ex:
                print('Error registering event handler to interface {0} : {1}\n'.format(interface_id, ex))

        self.interface_event_handlers.clear()
        print('Event handler unregistered from interfaces ...')
        self.mutex.release()


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
    if not PySpin.IsReadable(node_gev_enumeration):
        print('EnumerateGEVInterfaces node is unreadable. Aborting...')
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

    # Create system event handler
    #
    # *** NOTES ***
    # The SystemEventHandler has been written to accept a system object in the constructor
    # in order to register/unregister events to/from the system object
    system_event_handler = SystemEventHandler(system)

    # Register system event to the system
    #
    # *** NOTES ***
    # A system event is merely a combination of an interface arrival and an interface removal event.
    # This feature is currently only supported for GEV interface arrivals and removals.
    #
    # *** LATER ***
    # Interface arrival and removal events must all be unregistered manually.
    # This must be done prior to releasing the system and while they are still in scope.
    system.RegisterEventHandler(system_event_handler)

    system_event_handler.register_interface_event_to_system()
    system_event_handler.register_all_interface_events()

    # Wait for user to plug in and/or remove camera devices
    input('\nReady! Remove/Plug in cameras to test or press Enter to exit...\n')

    system_event_handler.unregister_all_interface_events()
    system_event_handler.unregister_interface_event_to_system()

    # Unregister system event handler from system object
    #
    # *** NOTES ***
    # It is important to unregister all arrival, removal, and interface event handlers
    # registered to the system.
    system.UnregisterEventHandler(system_event_handler)

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
