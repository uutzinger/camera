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
# StereoAcquisition.cpp shows how to acquire image sets from a stereo
# camera. The image sets are then saved to file and/or used to compute 3D
# point cloud and saved as a PLY (Polygon File Format) file.
#
# This example touches on the preparation and cleanup of a camera just before
# and just after the acquisition of images. Image retrieval and conversion,
# grabbing image data, and saving images are all covered as well.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import os
import PySpin
import sys
import platform


def set_selector_to_value(nodemap, selector, value):
    """
    Set a selector node to a specified value in a node map.

    :param nodemap: The node map from the camera.
    :param selector: The name of the selector node.
    :param value: The value to set for the selector.
    :type nodemap: INodeMap
    :type selector: string
    :type value: string
    :return: True if the selector is successfully set, False otherwise.
    :rtype: bool
    """
    try:
        selector_node = PySpin.CEnumerationPtr(nodemap.GetNode(selector))
        selector_entry = selector_node.GetEntryByName(value)
        selector_value = selector_entry.GetValue()
        selector_node.SetIntValue(selector_value)
        return True
    except PySpin.SpinnakerException:
        print("Failed to set {} selector to {} value".format(selector, value))
        return False


def enable_node(nodemap, node_name):
    """
    Enable a checkbox or boolean node in SpinView.

    :param nodemap: The node map from the camera.
    :param node_name: The name of the node to enable.
    :type nodemap: INodeMap
    :type node_name: string
    :return: True if the node was successfully enabled, False otherwise.
    :rtype: bool
    """
    try:
        the_node = PySpin.CBooleanPtr(nodemap.GetNode(node_name))
        if not PySpin.IsAvailable(the_node):
            print(f"Node '{node_name}' is not available.")
            return False

        if the_node.GetValue():
            return True
        elif PySpin.IsWritable(the_node):
            the_node.SetValue(True)
            return True
        else:
            print(f"Node '{node_name}' is not writable.")
            return False
    except PySpin.SpinnakerException as ex:
        print(f"Failed to enable node '{node_name}': {ex}")
        return False


def get_node_value(nodemap, node_name, data_type='float'):
    """
    Retrieve a node value from the camera's nodemap.

    :param nodemap: Nodemap object for the camera.
    :param node_name: The name of the node to retrieve.
    :param data_type: The expected data type of the node's value.
                         Supported values are 'float', 'int', 'bool'.
    :type nodemap: INodeMap
    :type node_name: string
    :type data_type: string 
    :return: The value of the node if successful, otherwise None.
    :rtype: float|int|bool|None
    """
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name)) if data_type == 'float' else (
           PySpin.CIntegerPtr(nodemap.GetNode(node_name)) if data_type == 'int' else
           PySpin.CBooleanPtr(nodemap.GetNode(node_name)))
    if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
        print(f"Node {node_name} not available or not readable.")
        return None
    if data_type == 'float':
        return node.GetValue()
    elif data_type == 'int':
        return node.GetValue()
    elif data_type == 'bool':
        return node.GetValue()
    else:
        print(f"Unsupported data type for node {node_name}.")
        return None

def configure_stereo_params(nodemap, stereo_params):
    """
    Configure stereo camera parameters by reading corresponding nodes from the nodemap.

    :param nodemap: Nodemap object from which to read parameters.
    :param stereo_params: The stereo parameters object to fill.
    :type nodemap: INodeMap
    :type stereo_params: StereoCameraParameters
    """
    coord_offset = get_node_value(nodemap, 'Scan3dCoordinateOffset', 'float')
    if coord_offset is not None:
        stereo_params.coordinateOffset = coord_offset
    else:
        print("Failed to retrieve Scan3dCoordinateOffset.")

    baseline = get_node_value(nodemap, 'Scan3dBaseline', 'float')
    if baseline is not None:
        stereo_params.baseline = baseline
    else:
        print("Failed to retrieve Scan3DBaseline.")

    focalLength = get_node_value(nodemap, 'Scan3dFocalLength', 'float')
    if focalLength is not None:
        stereo_params.focalLength = focalLength
    else:
        print("Failed to retrieve Scan3dFocalLength.")

    principal_point_u = get_node_value(nodemap, 'Scan3dPrincipalPointU', 'float')
    if principal_point_u is not None:
        stereo_params.principalPointU = principal_point_u
    else:
        print("Failed to retrieve Scan3dPrincipalPointU.")

    principal_point_v = get_node_value(nodemap, 'Scan3dPrincipalPointV', 'float')
    if principal_point_v is not None:
        stereo_params.principalPointV = principal_point_v
    else:
        print("Failed to retrieve Scan3dPrincipalPointV.")

    disparity_scale_factor = get_node_value(nodemap, 'Scan3dCoordinateScale', 'float')
    if disparity_scale_factor is not None:
        stereo_params.disparityScaleFactor = disparity_scale_factor
    else:
        print("Failed to retrieve Scan3dCoordinateScale.")

    invalid_data_value = get_node_value(nodemap, 'Scan3dInvalidDataValue', 'float')
    if invalid_data_value is not None:
        stereo_params.invalidDataValue = invalid_data_value
    else:
        print("Failed to retrieve Scan3dInvalidDataValue.")

    invalid_data_flag = get_node_value(nodemap, 'Scan3dInvalidDataFlag', 'bool')
    if invalid_data_flag is not None:
        stereo_params.invalidDataFlag = invalid_data_flag
    else:
        print("Failed to retrieve Scan3dInvalidDataFlag.")

def configure_chunk_data(nodemap):
    """
    This function configures the camera to add chunk data to each image. It does
    this by enabling each type of chunk data before enabling chunk data mode.
    When chunk data is turned on, the data is made available in both the nodemap
    and each image.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise
    :rtype: bool
    """
    try:
        result = True
        print('\n*** CONFIGURING CHUNK DATA ***\n')

        # Activate chunk mode
        #
        # *** NOTES ***
        # Once enabled, chunk data will be available at the end of the payload
        # of every image captured until it is disabled. Chunk data can also be
        # retrieved from the nodemap.
        chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))

        if PySpin.IsWritable(chunk_mode_active):
            chunk_mode_active.SetValue(True)

        print('Chunk mode activated...')

        # Enable all types of chunk data
        #
        # *** NOTES ***
        # Enabling chunk data requires working with nodes: "ChunkSelector"
        # is an enumeration selector node and "ChunkEnable" is a boolean. It
        # requires retrieving the selector node (which is of enumeration node
        # type), selecting the entry of the chunk data to be enabled, retrieving
        # the corresponding boolean, and setting it to be true.
        #
        # In this example, all chunk data is enabled, so these steps are
        # performed in a loop. Once this is complete, chunk mode still needs to
        # be activated.
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))

        if not PySpin.IsReadable(chunk_selector) or not PySpin.IsWritable(chunk_selector):
            print('Unable to retrieve chunk selector. Aborting...\n')
            return False

        # Retrieve entries
        #
        # *** NOTES ***
        # PySpin handles mass entry retrieval in a different way than the C++
        # API. Instead of taking in a NodeList_t reference, GetEntries() takes
        # no parameters and gives us a list of INodes. Since we want these INodes
        # to be of type CEnumEntryPtr, we can use a list comprehension to
        # transform all of our collected INodes into CEnumEntryPtrs at once.
        entries = [PySpin.CEnumEntryPtr(chunk_selector_entry) for chunk_selector_entry in chunk_selector.GetEntries()]

        print('Enabling entries...')

        # Iterate through our list and select each entry node to enable
        for chunk_selector_entry in entries:
            # Go to next node if problem occurs
            if not PySpin.IsReadable(chunk_selector_entry):
                continue

            chunk_selector.SetIntValue(chunk_selector_entry.GetValue())

            chunk_str = '\t {}:'.format(chunk_selector_entry.GetSymbolic())

            # Retrieve corresponding boolean
            chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable'))

            # Enable the boolean, thus enabling the corresponding chunk data
            if not PySpin.IsAvailable(chunk_enable):
                print('{} not available'.format(chunk_str))
                result = False
            elif chunk_enable.GetValue() is True:
                print('{} enabled'.format(chunk_str))
            elif PySpin.IsWritable(chunk_enable):
                chunk_enable.SetValue(True)
                print('{} enabled'.format(chunk_str))
            else:
                print('{} not writable'.format(chunk_str))
                result = False

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def disable_chunk_data(nodemap):
    """
    This function disables each type of chunk data before disabling chunk data mode.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise
    :rtype: bool
    """
    try:
        result = True

        # Retrieve the selector node
        chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))

        if not PySpin.IsReadable(chunk_selector) or not PySpin.IsWritable(chunk_selector):
            print('Unable to retrieve chunk selector. Aborting...\n')
            return False

        # Retrieve entries
        #
        # *** NOTES ***
        # PySpin handles mass entry retrieval in a different way than the C++
        # API. Instead of taking in a NodeList_t reference, GetEntries() takes
        # no parameters and gives us a list of INodes. Since we want these INodes
        # to be of type CEnumEntryPtr, we can use a list comprehension to
        # transform all of our collected INodes into CEnumEntryPtrs at once.
        entries = [PySpin.CEnumEntryPtr(chunk_selector_entry) for chunk_selector_entry in chunk_selector.GetEntries()]

        print('Disabling entries...')

        for chunk_selector_entry in entries:
            # Go to next node if problem occurs
            if not PySpin.IsReadable(chunk_selector_entry):
                continue

            chunk_selector.SetIntValue(chunk_selector_entry.GetValue())

            chunk_symbolic_form = '\t {}:'.format(chunk_selector_entry.GetSymbolic())

            # Retrieve corresponding boolean
            chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable'))

            # Disable the boolean, thus disabling the corresponding chunk data
            if not PySpin.IsAvailable(chunk_enable):
                print('{} not available'.format(chunk_symbolic_form))
                result = False
            elif not chunk_enable.GetValue():
                print('{} disabled'.format(chunk_symbolic_form))
            elif PySpin.IsWritable(chunk_enable):
                chunk_enable.SetValue(False)
                print('{} disabled'.format(chunk_symbolic_form))
            else:
                print('{} not writable'.format(chunk_symbolic_form))

        # Deactivate Chunk Mode
        chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))

        if not PySpin.IsWritable(chunk_mode_active):
            print('Unable to deactivate chunk mode. Aborting...\n')
            return False

        chunk_mode_active.SetValue(False)

        print('Chunk mode deactivated...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def display_chunk_data_from_image(image):
    """
    This function displays a select amount of chunk data from the image. Unlike
    accessing chunk data via the nodemap, there is no way to loop through all
    available data.

    :param image: Image to acquire chunk data from
    :type image: Image object
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('Printing chunk data from image...')
    try:
        result = True
        print(type(image))
        # Retrieve chunk data from image
        #
        # *** NOTES ***
        # When retrieving chunk data from an image, the data is stored in a
        # ChunkData object and accessed with getter functions.
        chunk_data = image.GetChunkData()

        # Retrieve frame ID
        frame_id = chunk_data.GetFrameID()
        print('\tFrame ID: {}'.format(frame_id))

        # The following are stereo-specific chunk data attributes
        # Retrieve stereo height; stereo height recorded in pixels
        stereo_height = chunk_data.GetStereoHeight()
        print('\tStereo Height: {}'.format(stereo_height))

        # Retrieve stereo width; stereo width recorded in pixels
        stereo_width = chunk_data.GetStereoWidth()
        print('\tStereo Width: {}'.format(stereo_width))

        # Retrieve Scan3dCoordinateOffset; Scan3dCoordinateOffset recorded in pixels
        scan3d_coordinate_offset = chunk_data.GetScan3dCoordinateOffset()
        print('\tScan3dCoordinateOffset: {}'.format(scan3d_coordinate_offset))

        # Retrieve Scan3dBaseline; Scan3dBaseline recorded in meters
        scan3d_baseline = chunk_data.GetScan3dBaseline()
        print('\tScan3dBaseline: {}'.format(scan3d_baseline))

        # Retrieve Scan3dFocalLength; Scan3dFocalLength recorded in pixels
        scan3d_focal_length = chunk_data.GetScan3dFocalLength()
        print('\tScan3dFocalLength: {}'.format(scan3d_focal_length))

        # Retrieve MinZ; MinZ recorded in meters
        min_z = chunk_data.GetMinZ()
        print('\tMinZ: {}'.format(min_z))

        # Retrieve MaxZ; MaxZ recorded in meters
        max_z = chunk_data.GetMaxZ()
        print('\tMaxZ: {}'.format(max_z))

        # Retrieve Scan3dPrincipalPointU; Scan3dPrincipalPointU recorded in pixels
        scan3d_principal_point_u = chunk_data.GetScan3dPrincipalPointU()
        print('\tScan3dPrincipalPointU: {}'.format(scan3d_principal_point_u))

        # Retrieve Scan3dPrincipalPointV; Scan3dPrincipalPointV recorded in pixels
        scan3d_principal_point_v = chunk_data.GetScan3dPrincipalPointV()
        print('\tScan3dPrincipalPointV: {}'.format(scan3d_principal_point_v))

        # Retrieve SmallPenalty; SmallPenalty recorded in pixels
        small_penalty = chunk_data.GetSmallPenalty()
        print('\tSmallPenalty: {}'.format(small_penalty))

        # Retrieve LargePenalty; LargePenalty recorded in pixels
        large_penalty = chunk_data.GetLargePenalty()
        print('\tLargePenalty: {}'.format(large_penalty))

        #Retrieve UniquenessRatio; UniquenessRatio recorded in percentage
        uniqueness_ratio = chunk_data.GetUniquenessRatio()
        print('\tUniquenessRatio: {}'.format(uniqueness_ratio))

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result

def compute_3d_point_cloud_and_save(stereo_params, image_list, filename_prefix, i):
    """
    This function computes a point cloud from a disparity image,
    and saves the point cloud into a .ply file

    :param stereo_params: the stereo parameters.
    :param image_list: list of aquired streams.
    :param filename_prefix: file name prefix.
    :param i: local index of the image list.
    :type stereo_params: StereoCameraParameters
    :type image_list: ImageList
    :type filename_prefix: string
    :type i: int
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    point_cloud_params = PySpin.PointCloudParameters()

    point_cloud_params.decimationFactor = 1
    point_cloud_params.ROIImageLeft = 0
    point_cloud_params.ROIImageTop = 0
    point_cloud_params.ROIImageRight = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_DISPARITY_SENSOR1).GetWidth()
    point_cloud_params.ROIImageBottom = image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_DISPARITY_SENSOR1).GetHeight()
    
    point_cloud = PySpin.ImageUtilityStereo.ComputePointCloud(
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_DISPARITY_SENSOR1),
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RECTIFIED_SENSOR1),
        point_cloud_params,
        stereo_params);

    filename = '%sPointCloud_%d.ply' % (filename_prefix, i)
    print('Save point cloud to file: %s\n' % filename)

    point_cloud.SavePointCloudAsPly(filename);

    return True;

def print_stereo_params(stereo_params):
    """
    This function prints the stereo parameters.

    :param stereo_params: the stereo parameters.
    :type stereo_params: StereoCameraParameters
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    
    print('coordinateOffset: ', stereo_params.coordinateOffset)
    print('baseline: ', stereo_params.baseline)
    print('principalPointU: ', stereo_params.principalPointU)
    print('principalPointV: ', stereo_params.principalPointV)
    print('disparityScaleFactor: ', stereo_params.disparityScaleFactor)
    print('invalidDataValue: ', stereo_params.invalidDataValue)
    print('invalidDataFlag: ', stereo_params.invalidDataFlag)

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
   
def save_images_to_files(stream_transmit_flags, image_list, filename_prefix, i):
    """
    This function saves an image list to files.

    :param stream_transmit_flags: list of boolean flags refering to the various streams indicating if to trasmit the stream.
    :param image_list: list of aquired streams.
    :param filename_prefix: file name prefix.
    :param i: local index of the image list.
    :type stream_transmit_flags: dict
    :type image_list: ImageList
    :type filename_prefix: string
    :type i: int
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    print('Save images to files.')

    if stream_transmit_flags['rawSensor1TransmitEnabled']:
        # StereoAcquisition_24102534_RectSensor1_0.png
        filename = '%sRawSensor1_%d.png' % (filename_prefix, i)
        print('Save raw Sensor1 image to file: %s' % filename)
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RAW_SENSOR1).Save(filename);

    if stream_transmit_flags['rawSensor2TransmitEnabled']:
        filename = '%sRawSensor2_%d.png' % (filename_prefix, i)
        print('Save raw Sensor2 image to file: %s' % filename)
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RAW_SENSOR2).Save(filename);

    if stream_transmit_flags['rectSensor1TransmitEnabled']:
        filename = '%sRectSensor1_%d.png' % (filename_prefix, i)
        print('Save rect Sensor1 image to file: %s' % filename)
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RECTIFIED_SENSOR1).Save(filename);

    if stream_transmit_flags['rectSensor2TransmitEnabled']:
        filename = '%sRectSensor2_%d.png' % (filename_prefix, i)
        print('Save rect Sensor2 image to file: %s' % filename)
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_RECTIFIED_SENSOR2).Save(filename);

    if stream_transmit_flags['disparityTransmitEnabled']:
        filename = '%sDisparity_%d.pgm' % (filename_prefix, i)
        print('Save disparity image to file: %s' % filename)
        image_list.GetByPayloadType(PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_DISPARITY_SENSOR1).Save(filename);

    return True


def enable_camera_stream(cam_node_map, source, component):
    """
    This function enables a camera stream.

    :param cam_node_map: Camera node map.
    :param source: source of the stream.
    :param component: component of the stream.
    :type cam_node_map: INodeMap
    :type source: string
    :type component: string
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # Selects the source (sensor1 or sensor2)
    if not set_selector_to_value(cam_node_map, "SourceSelector", source):
        return False

    # Selects the component (raw, rectified or disparity)
    if not set_selector_to_value(cam_node_map, "ComponentSelector", component):
        return False

    # Enables the stream
    if not enable_node(cam_node_map, 'ComponentEnable'):
        return False

    return True
    


def acquire_images(cam, nodemap, nodemap_tldevice, device_serial_number):
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

        
        stream_transmit_flags = {}
        stream_transmit_flags['rawSensor1TransmitEnabled'] = False
        stream_transmit_flags['rawSensor2TransmitEnabled'] = False
        stream_transmit_flags['rectSensor1TransmitEnabled'] = True
        stream_transmit_flags['rectSensor2TransmitEnabled'] = False
        stream_transmit_flags['disparityTransmitEnabled'] = True

        # Enable required streams
        if not enable_camera_stream(nodemap, 'Sensor1', 'Rectified') \
        or not enable_camera_stream(nodemap, 'Sensor1', 'Disparity'):
            # if either stream is not enabled
            print('Unable to enable required streams. Aborting...')
            return False
            
        #  Begin acquiring images
        #
        #  *** NOTES ***
        #  What happens when the camera begins acquiring images depends on the
        #  acquisition mode. Single frame captures only a single image, multi
        #  frame catures a set number of images, and continuous captures a
        #  continuous stream of images. Because the example calls for the
        #  retrieval of 3 images, continuous mode has been set.
        #
        #  *** LATER ***
        #  Image acquisition must be ended when no more images are needed.

        cam.BeginAcquisition()
        print('Acquiring images...')


        postProcessDisparity = True
        doComputePointCloud = True
        
        stereo_params = PySpin.StereoCameraParameters()
        configure_stereo_params(nodemap, stereo_params)
        
        print('*** STEREO PARAMETERS *** ')
        print_stereo_params(stereo_params)
        
        timeoutInMilliSecs = 2000;
        num_images_sets = 3
        print('\nAcquiring %d image sets.\n' % num_images_sets)
        
        # Retrieve, process the images
        for i in range(num_images_sets):
            try:
                print('Acquiring stereo image set: %d' % i)

                image_list = cam.GetNextImageSync(timeoutInMilliSecs)

                if not validate_image_list(stream_transmit_flags, image_list):
                    print('Failed to get next image set.')
                    continue;

                # Chunk Data is only available from the last Image in the ImageList for multi-stream cameras
                img = image_list.GetByIndex(image_list.GetSize() - 1)
                display_chunk_data_from_image(img)

                for j, image in enumerate(image_list):
                    
                    if image.GetImagePayloadType() == PySpin.SPINNAKER_IMAGE_PAYLOAD_TYPE_DISPARITY_SENSOR1:
                        payload_type_string = 'Disparity'
                        if postProcessDisparity:
                            print('Applying SpeckleFilter on disparity image....')
                            disparity_img = PySpin.ImageUtilityStereo.FilterSpeckles(image,
                                                                                       40,
                                                                                       4,
                                                                                       stereo_params.disparityScaleFactor,
                                                                                       stereo_params.invalidDataValue)

                filename_prefix = 'StereoAcquisition_%s_' % (device_serial_number)
                if not save_images_to_files(stream_transmit_flags, image_list, filename_prefix, i):
                    print('Failed to save images.')
                    result = False
                    break
                    
                if (doComputePointCloud and stream_transmit_flags['disparityTransmitEnabled'] and 
                    stream_transmit_flags['rectSensor1TransmitEnabled']):
                    if not compute_3d_point_cloud_and_save(stereo_params, image_list, filename_prefix, i):
                        print('Failed to compute the 3D point cloud.')
                        result = False
                        break

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
        
        # Configure chunk data
        print('Configuring chunk data...')
        result &= configure_chunk_data(nodemap)

        # Acquire images
        print('Acquiring images...')
        result &= acquire_images(cam, nodemap, nodemap_tldevice, device_serial_number)

        # Disable chunk data
        print('Disabling chunk data...')
        result &= disable_chunk_data(nodemap)

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
