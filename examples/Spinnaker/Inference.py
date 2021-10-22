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
# Inference.py shows how to perform the following:
#   - Upload custom inference neural networks to the camera (DDR or Flash)
#   - Inject sample test image
#   - Enable/Configure chunk data
#   - Enable/Configure trigger inference ready sync
#   - Acquire images
#   - Display inference data from acquired image chunk data
#   - Disable previously configured camera configurations
#
# Inference is only available for Firefly deep learning cameras.
# See the related content section on the Firefly DL product page for relevant
# documentation.
# https://www.flir.com/products/firefly-dl/
# It can also be helpful to familiarize yourself with the Acquisition,
# ChunkData and FileAccess_QuickSpin examples.

import PySpin
import numpy as np
import os
import sys
from enum import Enum

# Use the following enum and global constant to select whether inference network
# type is Detection or Classification.

class InferenceNetworkType(Enum):
    # This network determines the  most likely class given a set of predetermined,
    # trained options. Object detection can also provide a location within the
    # image (in the form of a "bounding box" surrounding the class), and can
    # detect multiple objects.
    DETECTION = 1
    # This network determines the best option from a list of predetermined options;
    # the camera gives a percentage that determines the likelihood of the currently
    # perceived image being one of the classes it has been trained to recognize.
    CLASSIFICATION = 2

CHOSEN_INFERENCE_NETWORK_TYPE = InferenceNetworkType.DETECTION

# Use the following enum and global constant to select whether uploaded inference
# network and injected image should be written to camera flash or DDR
class FileUploadPersistence(Enum):
    FLASH = 1  # Slower upload but data persists after power cycling the camera
    DDR = 2    # Faster upload but data clears after power cycling the camera

CHOSEN_FILE_UPLOAD_PERSISTENCE = FileUploadPersistence.DDR

# The example provides two existing custom networks that can be uploaded
# on to the camera to demonstrate classification and detection capabilities.
# "Network_Classification" file is created with Tensorflow using a mobilenet
# neural network for classifying flowers.
# "Network_Detection" file is created with Caffe using mobilenet SSD network
# for people object detection.
# Note: Make sure these files exist on the system and are accessible by the example
NETWORK_FILE_PATH = ("Network_Classification" if ((CHOSEN_INFERENCE_NETWORK_TYPE) \
                                            == InferenceNetworkType.CLASSIFICATION) \
                                            else "Network_Detection")

# The example provides two raw images that can be injected into the camera
# to demonstrate camera inference classification and detection capabilities. Jpeg
# representation of the raw images can be found packaged with the example with
# the names "Injected_Image_Classification_Daisy.jpg" and "Injected_Image_Detection_Aeroplane.jpg".
# Note: Make sure these files exist on the system and are accessible by the example
INJECTED_IMAGE_FILE_PATH = ("Injected_Image_Classification.raw" if ((CHOSEN_INFERENCE_NETWORK_TYPE) \
                                            == InferenceNetworkType.CLASSIFICATION) \
                                            else "Injected_Image_Detection.raw")

# The injected images have different ROI sizes so the camera needs to be
# configured to the appropriate width and height to match the injected image
INJECTED_IMAGE_WIDTH = 640 if CHOSEN_INFERENCE_NETWORK_TYPE == InferenceNetworkType.CLASSIFICATION else 720
INJECTED_IMAGE_HEIGHT = 400 if CHOSEN_INFERENCE_NETWORK_TYPE == InferenceNetworkType.CLASSIFICATION else 540

# Use the following enum to represent the inference bounding box type
class InferenceBoundingBoxType(Enum):
    INFERENCE_BOX_TYPE_RECTANGLE = 0
    INFERENCE_BOX_TYPE_CIRCLE = 1
    INFERENCE_BOX_TYPE_ROTATED_RECTANGLE = 2

# The sample classification inference network file was trained with the following
# data set labels
# Note: This list should match the list of labels used during the training
#       stage of the network file
LABEL_CLASSIFICATION = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# The sample detection inference network file was trained with the following
# data set labels
# Note: This list should match the list of labels used during the training
#       stage of the network file
LABEL_DETECTION = ["background", "aeroplane", "bicycle",     "bird",  "boat",        "bottle", "bus",
                    "car",        "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",
                    "motorbike",  "person",    "pottedplant", "sheep", "sofa",        "train",  "monitor"]

# This function prints the device information of the camera from the transport
# layer; please see NodeMapInfo example for more in-depth comments on printing
# device information from the nodemap.
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
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

# This function executes a file delete operation on the camera.
def camera_delete_file(nodemap):
    ptr_file_size = PySpin.CIntegerPtr(nodemap.GetNode("FileSize"))
    if not PySpin.IsReadable(ptr_file_size):
        print('Unable to query FileSize. Aborting...')
        return False
    
    if ptr_file_size.GetValue() == 0:
        # No file uploaded yet. Skip delete
        print('No files found, skipping file deletion.')
        return True

    print('Deleting file...')
    try:
        ptr_file_operation_selector = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationSelector"))
        if not PySpin.IsWritable(ptr_file_operation_selector):
            print('Unable to configure FileOperationSelector. Aborting...')
            return False

        ptr_file_operation_delete = PySpin.CEnumEntryPtr(ptr_file_operation_selector.GetEntryByName("Delete"))
        if not PySpin.IsReadable(ptr_file_operation_delete):
            print('Unable to configure FileOperationSelector Delete. Aborting...')
            return False
        
        ptr_file_operation_selector.SetIntValue(int(ptr_file_operation_delete.GetNumericValue()))
        
        ptr_file_operation_execute = PySpin.CCommandPtr(nodemap.GetNode("FileOperationExecute"))
        if not PySpin.IsWritable(ptr_file_operation_execute):
            print('Unable to configure FileOperationExecute. Aborting...')
            return False
        
        ptr_file_operation_execute.Execute()

        ptr_file_operation_status = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationStatus"))
        if not PySpin.IsReadable(ptr_file_operation_status):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        ptr_file_operation_status_success = PySpin.CEnumEntryPtr(ptr_file_operation_status.GetEntryByName("Success"))
        if not PySpin.IsReadable(ptr_file_operation_status_success):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        if ptr_file_operation_status.GetCurrentEntry().GetNumericValue() != ptr_file_operation_status_success.GetNumericValue():
            print('Failed to delete file! File Operation Status : %s' %ptr_file_operation_status.GetCurrentEntry().GetSymbolic())
            return False
        
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False
    
    return True

# This function executes file open/write on the camera, sets the uploaded file persistence
# and attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write.
def camera_open_file(nodemap):
    print('Opening file for writing...')
    try:
        ptr_file_operation_selector = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationSelector"))
        if not PySpin.IsWritable(ptr_file_operation_selector):
            print('Unable to configure FileOperationSelector. Aborting...')
            return False

        ptr_file_operation_open = PySpin.CEnumEntryPtr(ptr_file_operation_selector.GetEntryByName("Open"))
        if not PySpin.IsReadable(ptr_file_operation_open):
            print('Unable to configure FileOperationSelector Open. Aborting...')
            return False
        
        ptr_file_operation_selector.SetIntValue(int(ptr_file_operation_open.GetNumericValue()))

        ptr_file_open_mode = PySpin.CEnumerationPtr(nodemap.GetNode("FileOpenMode"))
        if not PySpin.IsWritable(ptr_file_open_mode):
            print('Unable to configure ptr_file_open_mode. Aborting...')
            return False

        ptr_file_open_mode_write = PySpin.CEnumEntryPtr(ptr_file_open_mode.GetEntryByName("Write"))
        if not PySpin.IsReadable(ptr_file_open_mode_write):
            print('Unable to configure FileOperationSelector Write. Aborting...')
            return False       
        
        ptr_file_open_mode.SetIntValue(int(ptr_file_open_mode_write.GetNumericValue()))
        
        ptr_file_operation_execute = PySpin.CCommandPtr(nodemap.GetNode("FileOperationExecute"))
        if not PySpin.IsWritable(ptr_file_operation_execute):
            print('Unable to configure FileOperationExecute. Aborting...')
            return False
        
        ptr_file_operation_execute.Execute()

        ptr_file_operation_status = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationStatus"))
        if not PySpin.IsReadable(ptr_file_operation_status):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        ptr_file_operation_status_success = PySpin.CEnumEntryPtr(ptr_file_operation_status.GetEntryByName("Success"))
        if not PySpin.IsReadable(ptr_file_operation_status_success):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        if ptr_file_operation_status.GetCurrentEntry().GetNumericValue() != ptr_file_operation_status_success.GetNumericValue():
            print('Failed to open file for writing! File Operation Status : %s' %ptr_file_operation_status.GetCurrentEntry().GetSymbolic())
            return False

        # Set file upload persistence settings
        ptr_file_write_to_flash = PySpin.CBooleanPtr(nodemap.GetNode("FileWriteToFlash"))
        if PySpin.IsWritable(ptr_file_write_to_flash):
            if CHOSEN_FILE_UPLOAD_PERSISTENCE == FileUploadPersistence.FLASH:
                ptr_file_write_to_flash.SetValue(True)
                print('FileWriteToFlash is set to true')
            else:
                ptr_file_write_to_flash.SetValue(False)
                print('FileWriteToFlash is set to false')

        # Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
        ptr_file_access_length = PySpin.CIntegerPtr(nodemap.GetNode("FileAccessLength"))
        if not PySpin.IsReadable(ptr_file_access_length) or not PySpin.IsWritable(ptr_file_access_length):
            print('Unable to query/configure FileAccessLength. Aborting...')
            return False

        # Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
        ptr_file_access_buffer = PySpin.CRegisterPtr(nodemap.GetNode("FileAccessBuffer"))
        if not PySpin.IsReadable(ptr_file_access_buffer):
            print('Unable to query FileAccessBuffer. Aborting...')
            return False

        if ptr_file_access_length.GetValue() < ptr_file_access_buffer.GetLength():
            try:
                ptr_file_access_length.SetValue(ptr_file_access_buffer.GetLength())
            except PySpin.SpinnakerException as ex:
                print('Unexpected exception: %s' % ex)
        
        # Set File Access Offset to zero 
        ptr_file_access_offset = PySpin.CIntegerPtr(nodemap.GetNode("FileAccessOffset"))
        if not PySpin.IsReadable(ptr_file_access_offset) or not PySpin.IsWritable(ptr_file_access_offset):
             print('Unable to query/configure ptrFileAccessOffset. Aborting...')
             return False
        ptr_file_access_offset.SetValue(0)

    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return True
    
# This function executes a file write operation on the camera.
def camera_write_to_file(nodemap):
    try:
        ptr_file_operation_selector = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationSelector"))
        if not PySpin.IsWritable(ptr_file_operation_selector):
            print('Unable to configure FileOperationSelector. Aborting...')
            return False 
        
        ptr_file_operation_write = PySpin.CEnumEntryPtr(ptr_file_operation_selector.GetEntryByName("Write"))
        if not PySpin.IsReadable(ptr_file_operation_write):
            print('Unable to configure FileOperationSelector Write. Aborting...')
            return False
        
        ptr_file_operation_selector.SetIntValue(int(ptr_file_operation_write.GetNumericValue()))

        ptr_file_operation_execute = PySpin.CCommandPtr(nodemap.GetNode("FileOperationExecute"))
        if not PySpin.IsWritable(ptr_file_operation_execute):
            print('Unable to configure FileOperationExecute. Aborting...')
            return False

        ptr_file_operation_execute.Execute()

        ptr_file_operation_status = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationStatus"))
        if not PySpin.IsReadable(ptr_file_operation_status):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        ptr_file_operation_status_success = PySpin.CEnumEntryPtr(ptr_file_operation_status.GetEntryByName("Success"))
        if not PySpin.IsReadable(ptr_file_operation_status_success):
            print('Unable to query FileOperationStatus Success. Aborting...')
            return False

        if ptr_file_operation_status.GetCurrentEntry().GetNumericValue() != ptr_file_operation_status_success.GetNumericValue():
            print('Failed to write to file! File Operation Status : %s' %ptr_file_operation_status.GetCurrentEntry().GetSymbolic())
            return False

    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False
    
    return True

# This function executes a file close operation on the camera.
def camera_close_file(nodemap):
    print('Closing file...')
    try:
        ptr_file_operation_selector = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationSelector"))
        if not PySpin.IsWritable(ptr_file_operation_selector):
            print('Unable to configure FileOperationSelector. Aborting...')
            return False

        ptr_file_operation_close = PySpin.CEnumEntryPtr(ptr_file_operation_selector.GetEntryByName("Close"))
        if not PySpin.IsReadable(ptr_file_operation_close):
            print('Unable to configure FileOperationSelector Close. Aborting...')
            return False       

        ptr_file_operation_selector.SetIntValue(int(ptr_file_operation_close.GetNumericValue()))

        ptr_file_operation_execute = PySpin.CCommandPtr(nodemap.GetNode("FileOperationExecute"))
        if not PySpin.IsWritable(ptr_file_operation_execute):
            print('Unable to configure FileOperationExecute. Aborting...')
            return False
        
        ptr_file_operation_execute.Execute()

        ptr_file_operation_status = PySpin.CEnumerationPtr(nodemap.GetNode("FileOperationStatus"))
        if not PySpin.IsReadable(ptr_file_operation_status):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        ptr_file_operation_status_success = PySpin.CEnumEntryPtr(ptr_file_operation_status.GetEntryByName("Success"))
        if not PySpin.IsReadable(ptr_file_operation_status_success):
            print('Unable to query FileOperationStatus. Aborting...')
            return False

        if ptr_file_operation_status.GetCurrentEntry().GetNumericValue() != ptr_file_operation_status_success.GetNumericValue():
            print('Failed to close the file! File Operation Status : %s' %ptr_file_operation_status.GetCurrentEntry().GetSymbolic())
            return False

    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False
    
    return True  

# This function uploads a file on the system to the camera given the selected
# file selector entry.
def upload_file_to_camera(nodemap, file_selector_entry_name, file_path):
    print('\n*** CONFIGURING FILE SELECTOR ***')

    ptr_file_selector = PySpin.CEnumerationPtr(nodemap.GetNode('FileSelector'))
    if not PySpin.IsWritable(ptr_file_selector):
        print('Unable to configure FileSelector. Aborting...')
        return False
    
    ptr_inference_selector_entry = PySpin.CEnumEntryPtr(ptr_file_selector.GetEntryByName(file_selector_entry_name))
    if not PySpin.IsReadable(ptr_inference_selector_entry):
        print('Unable to query FileSelector entry %s ' %file_selector_entry_name + '. Aborting...')
        return False

    # Set file selector to entry
    print('Setting FileSelector to %s ' %ptr_inference_selector_entry.GetSymbolic() + '...\n')
    ptr_file_selector.SetIntValue(int(ptr_inference_selector_entry.GetNumericValue()))

    # Delete file on camera before writing in case camera runs out of space
    if camera_delete_file(nodemap) != True:
        print('Failed to delete existing file for selector entry %s' %ptr_inference_selector_entry.GetSymbolic() + '. Aborting...')
        return False
    
    # Open file on camera for write
    if camera_open_file(nodemap) != True:
        if not camera_close_file(nodemap):
            print('Problem opening file node. Aborting...')
            return False
        if not camera_open_file(nodemap):
            print('Problem opening file node. Aborting...')
            return False

    # check node
    ptr_file_access_length = PySpin.CIntegerPtr(nodemap.GetNode('FileAccessLength'))
    if not PySpin.IsReadable(ptr_file_access_length) or not PySpin.IsWritable(ptr_file_access_length):
        print('Unable to query FileAccessLength. Aborting...')
        return False

    ptr_file_access_buffer = PySpin.CRegisterPtr(nodemap.GetNode('FileAccessBuffer'))
    if not PySpin.IsReadable(ptr_file_access_buffer) or not PySpin.IsWritable(ptr_file_access_buffer):
        print('Unable to query FileAccessBuffer. Aborting...')
        return False
    
    ptr_file_access_offset = PySpin.CIntegerPtr(nodemap.GetNode('FileAccessOffset'))
    if not PySpin.IsReadable(ptr_file_access_offset) or not PySpin.IsWritable(ptr_file_access_offset):
        print('Unable to query FileAccessOffset. Aborting...')
        return False
    
    ptr_file_access_result = PySpin.CIntegerPtr(nodemap.GetNode('FileOperationResult'))
    if not PySpin.IsReadable(ptr_file_access_result):
        print('Unable to query FileOperationResult. Aborting...')
        return False

    # Load network file from path depending on network type
    with open(file_path, 'rb') as fd:
        fd.seek(0, os.SEEK_END)
        num_bytes = fd.tell()
        fd.seek(0,0)
        file_bytes = np.fromfile(fd, dtype=np.ubyte, count=num_bytes)

        if len(file_bytes) == 0:
            print('Failed to load file path : %s' %file_path + '. Aborting...')
            return False

        total_bytes_to_write = len(file_bytes)
        intermediate_buffer_size = ptr_file_access_length.GetValue()
        write_iterations = (total_bytes_to_write // intermediate_buffer_size) + \
                           (0 if ((total_bytes_to_write % intermediate_buffer_size) == 0) else 1)

        if total_bytes_to_write == 0:
            print('Empty Image. No data will be written to camera. Aborting...')
            return False
        
        print('Start uploading %s' %file_path + ' to device...')

        print('Total bytes to write: %s' % total_bytes_to_write)
        print('FileAccessLength: %s' % intermediate_buffer_size)
        print('Write iterations: %s' % write_iterations)

        bytes_left_to_write = total_bytes_to_write
        total_bytes_written = 0

        print('Writing data to device...')

        # Splitting the file into equal chunks (except the last chunk)
        sections = []
        for index in range(write_iterations):
            num = index * intermediate_buffer_size
            if num == 0:
                continue
            sections.append(num)
        split_data = np.array_split(file_bytes, sections)

        # Writing split data to camera 
        for i in range(write_iterations):
            # Set up data to write
            tmp_buffer = split_data[i]

            # Write to AccessBufferNode
            ptr_file_access_buffer.Set(tmp_buffer)

            if intermediate_buffer_size > bytes_left_to_write:
                ptr_file_access_length.SetValue(bytes_left_to_write)

            # Perform Write command
            if not camera_write_to_file(nodemap):
                print('Writing to stream failed. Aborting...')
                return False
            
            # Verify size of bytes written
            size_written = ptr_file_access_result.GetValue()

            # Keep track of total bytes written
            total_bytes_written += size_written

            # Keep track of bytes left to write
            bytes_left_to_write = total_bytes_to_write - total_bytes_written

            sys.stdout.write('\r')
            sys.stdout.write('Progress: %s' % int((i*100 / write_iterations)) + '%' )
            sys.stdout.flush()
        
        print('\nWriting complete')

        if not camera_close_file(nodemap):
            print('Failed to close file!')

    return True

# This function deletes the file uploaded to the camera given the selected
# file selector entry.
def delete_file_on_camera(nodemap, file_selector_entry_name):
    print('\n*** CLEANING UP FILE SELECTOR **')

    ptr_file_selector = PySpin.CEnumerationPtr(nodemap.GetNode("FileSelector"))
    if not PySpin.IsWritable(ptr_file_selector):
        print('Unable to configure FileSelector. Aborting...')
        return False
    
    ptr_inference_selector_entry = PySpin.CEnumEntryPtr(ptr_file_selector.GetEntryByName(file_selector_entry_name))
    if not PySpin.IsReadable(ptr_inference_selector_entry):
        print('Unable to query FileSelector entry ' + file_selector_entry_name + '. Aborting...')
        return False

    # Set file Selector entry 
    print('Setting FileSelector to %s ' %ptr_inference_selector_entry.GetSymbolic() + '...\n')
    ptr_file_selector.SetIntValue(int(ptr_inference_selector_entry.GetNumericValue()))

    if camera_delete_file(nodemap) != True:
        print('Failed to delete existing file for selector entry')
        return False

    return True

# This function enables or disables the given chunk data type based on
# the specified entry name.
def set_chunk_enable(nodemap, entry_name, enable):
    result = True
    ptr_chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))
    
    ptr_entry = PySpin.CEnumEntryPtr(ptr_chunk_selector.GetEntryByName(entry_name))
    if not PySpin.IsReadable(ptr_entry):
        print('Unable to find ' + entry_name + ' in ChunkSelector...')
        return False

    ptr_chunk_selector.SetIntValue(ptr_entry.GetValue())

    # Enable the boolean, thus enabling the corresponding chunk data
    print('Enabling ' + entry_name + '...')
    ptr_chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode("ChunkEnable"))
    if not PySpin.IsAvailable(ptr_chunk_enable):
        print('not available')
        return False
    
    if enable:
        if ptr_chunk_enable.GetValue():
            print('enabled')
        elif PySpin.IsWritable(ptr_chunk_enable):
            ptr_chunk_enable.SetValue(True)
            print('enabled')
        else:
            print('not writable')
            result = False
    else:
        if not ptr_chunk_enable.GetValue():
            print('disabled')
        elif PySpin.IsWritable(ptr_chunk_enable):
            ptr_chunk_enable.SetValue(False)
            print('disabled')
        else:
            print('not writable')
            result = False

    return result

# This function configures the camera to add inference chunk data to each image.
# When chunk data is turned on, the data is made available in both the nodemap
# and each image.
def configure_chunk_data(nodemap):
    result = True
    print('\n*** CONFIGURING CHUNK DATA ***')

    try:
        # Activate chunk mode
        #
        # *** NOTES ***
        # Once enabled, chunk data will be available at the end of the payload
        # of every image captured until it is disabled. Chunk data can also be
        # retrieved from the nodemap.
        
        ptr_chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode("ChunkModeActive"))
        if not PySpin.IsWritable(ptr_chunk_mode_active):
            print('Unable to active chunk mode. Aborting...')
            return False
        
        ptr_chunk_mode_active.SetValue(True)
        print('Chunk mode activated...')

        # Enable inference related chunks in chunk data

        # Retrieve the chunk data selector node
        ptr_chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))
        if not PySpin.IsReadable(ptr_chunk_selector):
            print('Unable to retrieve chunk selector (enum retrieval). Aborting...')
            return False
        
        # Enable chunk data inference Frame Id
        result = set_chunk_enable(nodemap, "InferenceFrameId", True)
        if result == False:
            print("Unable to enable Inference Frame Id chunk data. Aborting...")
            return result
        
        if CHOSEN_INFERENCE_NETWORK_TYPE == InferenceNetworkType.DETECTION:
            # Detection network type

            # Enable chunk data inference bounding box
            result = set_chunk_enable(nodemap, "InferenceBoundingBoxResult", True)
            if result == False:
                print("Unable to enable Inference Bounding Box chunk data. Aborting...")
                return result
        else:
            # Enable chunk data inference result
            result = set_chunk_enable(nodemap, "InferenceResult", True)
            if result == False:
                print("Unable to enable Inference Result chunk data. Aborting...")
                return result

            # Enable chunk data inference confidence
            result = set_chunk_enable(nodemap, "InferenceConfidence", True)
            if result == False:
                print("Unable to enable Inference Confidence chunk data. Aborting...")
                return result
            
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return result

# This function disables each type of chunk data before disabling chunk data mode.
def disable_chunk_data(nodemap):
    print('\n*** DISABLING CHUNK DATA ***')

    result = True
    try:
        ptr_chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode("ChunkSelector"))

        if not PySpin.IsReadable(ptr_chunk_selector):
            print('Unable to retrieve chunk selector. Aborting...')
            return False

        result = set_chunk_enable(nodemap, "InferenceFrameId", False)
        if result == False:
            print('Unable to disable Inference Frame Id chunk data. Aborting...')
            return result

        if CHOSEN_INFERENCE_NETWORK_TYPE == InferenceNetworkType.DETECTION:
            # Detection network type 

            # Disable chunk data inference bounding box
            result = set_chunk_enable(nodemap, "InferenceBoundingBoxResult", False)
            if result == False:
                print('Unable to disable Inference Bounding Box chunk data. Aborting...')
                return result
        else:
            # Classification network type

            # Disable chunk data inference result
            result = set_chunk_enable(nodemap, "InferenceResult", False)
            if result == False:
                print('Unable to disable Inference Result chunk data. Aborting...')
                return result
            
            # Disable chunk data inference confidence
            result = set_chunk_enable(nodemap, "InferenceConfidence", False)
            if result == False:
                print('Unable to disable Inference Confidence chunk data. Aborting...')
                return result
        
        # Deactivate ChunkMode
        ptr_chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode("ChunkModeActive"))
        if not PySpin.IsWritable(ptr_chunk_mode_active):
            print('Unable to deactivate chunk mode. Aborting...')
            return False

        ptr_chunk_mode_active.SetValue(False)
        print('Chunk mode deactivated...')

        # Disable Inference
        ptr_inference_enable = PySpin.CBooleanPtr(nodemap.GetNode("InferenceEnable"))
        if not PySpin.IsWritable(ptr_inference_enable):
            print('Unable to disable inference. Aborting...')
            return False

        ptr_inference_enable.SetValue(False)
        print('Inference disabled...')
    
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return result 

# This function displays the inference-related chunk data from the image.
def display_chunk_data(image):
    result = True
    print('Printing chunk data from image...')

    try:
        chunk_data = image.GetChunkData()

        inference_frame_ID = chunk_data.GetInferenceFrameId()
        print('\tInference Frame ID: %s' % inference_frame_ID)
        
        if CHOSEN_INFERENCE_NETWORK_TYPE == InferenceNetworkType.DETECTION:
            box_result = chunk_data.GetInferenceBoundingBoxResult()
            box_count = box_result.GetBoxCount()

            print('\tInference Bounding Box Result:')
            if box_count == 0:
                print('\t No bounding box')
            
            for i in range(box_count):
                box = box_result.GetBoxAt(i)
                if box.boxType == InferenceBoundingBoxType.INFERENCE_BOX_TYPE_RECTANGLE.value:
                    print('\t\tBox {0}: Class {1} ({2}): - {3:.4f}% - {4} (X={5} Y={6} W={7} H={8})'
                        .format(i+1,
                                box.classId,
                                LABEL_DETECTION[box.classId] if box.classId < len(LABEL_DETECTION) else "N/A",
                                box.confidence * 100,
                                "Rectangle",
                                box.rect.topLeftXCoord,
                                box.rect.topLeftYCoord,
                                box.rect.bottomRightXCoord - box.rect.topLeftXCoord,
                                box.rect.bottomRightYCoord - box.rect.topLeftYCoord))
                elif box.boxType == InferenceBoundingBoxType.INFERENCE_BOX_TYPE_CIRCLE.value:
                    print('\t\tBox {0}: Class {1} ({2}): - {3:.4f}% - {4} (X={5} Y={6} R={7})'
                        .format(i+1,
                                box.classId,
                                LABEL_DETECTION[box.classId] if box.classId < len(LABEL_DETECTION) else "N/A",
                                box.confidence * 100,
                                "Circle",
                                box.rect.topLeftXCoord,
                                box.rect.topLeftYCoord,
                                box.circle.radius))
                elif box.boxType == InferenceBoundingBoxType.INFERENCE_BOX_TYPE_ROTATED_RECTANGLE.value:
                    print('\t\tBox {0}: Class {1} ({2}): - {3:.4f}% - {4} (X1={5} Y1={6} X2={7} Y2={8} angle={9})'
                        .format(i+1,
                                box.classId,
                                LABEL_DETECTION[box.classId] if box.classId < len(LABEL_DETECTION) else "N/A",
                                box.confidence * 100,
                                "Rotated Rectangle",
                                box.rotatedRect.topLeftXCoord,
                                box.rotatedRect.topLeftYCoord,
                                box.rotatedRect.bottomRightXCoord,
                                box.rotatedRect.bottomRightYCoord,
                                box.rotatedRect.rotationAngle))
                else:
                    print('\t\tBox {0}: Class {1} ({2}): - {3:.4f}% - {4})'
                        .format(i+1,
                                box.classId,
                                LABEL_DETECTION[box.classId] if box.classId < len(LABEL_DETECTION) else "N/A",
                                box.confidence * 100,
                                "Unknown bounding box type (not supported)"))
        else:
            inference_result = chunk_data.GetInferenceResult()
            print('\t Inference Result: %s' %inference_result, end = '')
            print(' (%s)' % LABEL_CLASSIFICATION[inference_result] if inference_result < len(LABEL_CLASSIFICATION) else "N/A")

            inference_confidence = chunk_data.GetInferenceConfidence()
            print('\t Inference Confidence: %.6f' %inference_confidence)

    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return result

# This function disables trigger mode on the camera.
def disable_trigger(nodemap):
    print('\n*** IMAGE ACQUISITION ***')

    try:
        ptr_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
        if not PySpin.IsWritable(ptr_trigger_mode):
            print('Unable to configure TriggerMode. Aborting...')
            return False
        
        ptr_trigger_off = PySpin.CEnumEntryPtr(ptr_trigger_mode.GetEntryByName("Off"))
        if not PySpin.IsReadable(ptr_trigger_off):
            print('Unable to query TriggerMode Off. Aborting...')
            return False
        
        print('Configure TriggerMode to ' + ptr_trigger_off.GetSymbolic())
        ptr_trigger_mode.SetIntValue(int(ptr_trigger_off.GetNumericValue()))
    
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return True

# This function configures camera to run in "inference sync" trigger mode.
def configure_trigger(nodemap):
    print('\n*** CONFIGURING TRIGGER ***')

    try:
        # Configure TriggerSelector 
        ptr_trigger_selector = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSelector"))
        if not PySpin.IsWritable(ptr_trigger_selector):
            print('Unable to configure TriggerSelector. Aborting...')
            return False
        
        ptr_frame_start = PySpin.CEnumEntryPtr(ptr_trigger_selector.GetEntryByName("FrameStart"))
        if not PySpin.IsReadable(ptr_frame_start):
            print('Unable to query TriggerSelector FrameStart. Aborting...')
            return False
        
        print('Configure TriggerSelector to ' + ptr_frame_start.GetSymbolic())
        ptr_trigger_selector.SetIntValue(int(ptr_frame_start.GetNumericValue()))

        # Configure TriggerSource 
        ptr_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerSource"))
        if not PySpin.IsWritable(ptr_trigger_source):
            print('Unable to configure TriggerSource. Aborting...')
            return False

        ptr_inference_ready = PySpin.CEnumEntryPtr(ptr_trigger_source.GetEntryByName("InferenceReady"))
        if not PySpin.IsReadable(ptr_inference_ready):
            print('Unable to query TriggerSource InferenceReady. Aborting...')
            return False

        print('Configure TriggerSource to ' + ptr_inference_ready.GetSymbolic())
        ptr_trigger_source.SetIntValue(int(ptr_inference_ready.GetNumericValue()))

        # Configure TriggerMode 
        ptr_trigger_mode = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerMode"))
        if not PySpin.IsWritable(ptr_trigger_mode):
            print('Unable to configure TriggerMode. Aborting...')
            return False
        
        ptr_trigger_on = PySpin.CEnumEntryPtr(ptr_trigger_mode.GetEntryByName("On"))
        if not PySpin.IsReadable(ptr_trigger_on):
            print('Unable to query TriggerMode On. Aborting...')
            return False

        print('Configure TriggerMode to ' + ptr_trigger_on.GetSymbolic())
        ptr_trigger_mode.SetIntValue(int(ptr_trigger_on.GetNumericValue()))

    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return True

# This function enables/disables inference on the camera and configures the inference network type
def configure_inference(nodemap, is_enabled):
    if is_enabled:
        print('\n*** CONFIGURING INFERENCE (' + ("DETECTION" if ((CHOSEN_INFERENCE_NETWORK_TYPE) \
                                            == InferenceNetworkType.DETECTION) \
                                            else 'CLASSIFICATION') + ') ***')
    else:
        print('\n*** DISABLING INFERENCE ***')

    try:
        if is_enabled:
            ptr_inference_network_type_selector = PySpin.CEnumerationPtr(nodemap.GetNode("InferenceNetworkTypeSelector"))
            if not PySpin.IsWritable(ptr_inference_network_type_selector):
                print('Unable to query InferenceNetworkTypeSelector. Aborting...')
                return False
            
            network_type_string = ("Detection" if CHOSEN_INFERENCE_NETWORK_TYPE == InferenceNetworkType.DETECTION 
                                    else "Classification")

            # Retrieve entry node from enumeration node
            ptr_inference_network_type = PySpin.CEnumEntryPtr(ptr_inference_network_type_selector.GetEntryByName(network_type_string))
            if not PySpin.IsReadable(ptr_inference_network_type):
                print('Unable to set inference network type to %s' %network_type_string + ' (entry retrieval). Aborting...')
                return False
            
            inference_network_value = ptr_inference_network_type.GetNumericValue()
            ptr_inference_network_type_selector.SetIntValue(int(inference_network_value))

            print('Inference network type set to' + network_type_string + '...')

        print(('Enabling' if is_enabled else 'Disabling') + ' inference...')
        ptr_inference_enable = PySpin.CBooleanPtr(nodemap.GetNode("InferenceEnable"))
        if not PySpin.IsWritable(ptr_inference_enable):
            print('Unable to enable inference. Aborting...')
            return False
        
        ptr_inference_enable.SetValue(is_enabled)
        print('Inference '+'enabled...' if is_enabled else 'disabled...')
    
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return True

# This function configures camera test pattern to make use of the injected test image for inference
def configure_test_pattern(nodemap, is_enabled):
    if is_enabled:
        print('\n*** CONFIGURING TEST PATTERN ***')
    else:
        print('\n*** DISABLING TEST PATTERN ***')

    try:
        # Set TestPatternGeneratorSelector to PipelineStart
        ptr_test_pattern_generator_selector = PySpin.CEnumerationPtr(nodemap.GetNode("TestPatternGeneratorSelector"))
        if not PySpin.IsWritable(ptr_test_pattern_generator_selector):
            print('Unable to query TestPatternGeneratorSelector. Aborting...')
            return False

        if is_enabled:
            ptr_test_pattern_generator_pipeline_start = PySpin.CEnumEntryPtr(ptr_test_pattern_generator_selector.GetEntryByName("PipelineStart"))
            if not PySpin.IsReadable(ptr_test_pattern_generator_pipeline_start):
                print('Unable to query TestPatternGeneratorSelector PipelineStart. Aborting...')
                return False
            
            ptr_test_pattern_generator_selector.SetIntValue(int(ptr_test_pattern_generator_pipeline_start.GetNumericValue()))
            print('TestPatternGeneratorSelector set to ' + ptr_test_pattern_generator_pipeline_start.GetSymbolic() + '...')

        else:
            ptr_test_pattern_generator_sensor = PySpin.CEnumEntryPtr(ptr_test_pattern_generator_selector.GetEntryByName("Sensor"))
            if not PySpin.IsReadable(ptr_test_pattern_generator_sensor):
                print('Unable to query TestPatternGeneratorSelector Sensor. Aborting...')
                return False
            
            ptr_test_pattern_generator_selector.SetIntValue(int(ptr_test_pattern_generator_sensor.GetNumericValue()))
            print('TestPatternGeneratorSelector set to ' + ptr_test_pattern_generator_sensor.GetSymbolic() + '...')

        # Set TestPattern to InjectedImage
        ptr_test_pattern = PySpin.CEnumerationPtr(nodemap.GetNode("TestPattern"))
        if not PySpin.IsWritable(ptr_test_pattern):
            print('Unable to query TestPattern. Aborting...')
            return False

        if is_enabled:
            ptr_injected_image = PySpin.CEnumEntryPtr(ptr_test_pattern.GetEntryByName("InjectedImage"))
            if not PySpin.IsReadable(ptr_injected_image):
                print('Unable to query TestPattern InjectedImage. Aborting...')
                return False
            
            ptr_test_pattern.SetIntValue(int(ptr_injected_image.GetNumericValue()))
            print('TestPattern set to ' + ptr_injected_image.GetSymbolic() + '...')
        else:
            ptr_test_pattern_off = PySpin.CEnumEntryPtr(ptr_test_pattern.GetEntryByName("Off"))
            if not PySpin.IsReadable(ptr_test_pattern_off):
                print('Unable to query TestPattern Off. Aborting...')
                return False
            
            ptr_test_pattern.SetIntValue(int(ptr_test_pattern_off.GetNumericValue()))
            print('TestPattern set to ' + ptr_test_pattern_off.GetSymbolic() + '...')
        
        if is_enabled:
            # The inject images have different ROI sizes so camera needs to be configured to the appropriate
            # injected width and height
            ptr_injected_width = PySpin.CIntegerPtr(nodemap.GetNode("InjectedWidth"))
            if not PySpin.IsWritable(ptr_injected_width):
                print('Unable to query InjectedWidth. Aborting...')
                return False
            
            ptr_injected_width.SetValue(INJECTED_IMAGE_WIDTH if is_enabled else ptr_injected_width.GetMax())

            ptr_injected_height = PySpin.CIntegerPtr(nodemap.GetNode("InjectedHeight"))
            if not PySpin.IsWritable(ptr_injected_height):
                print('Unable to query InjectedHeight. Aborting...')
                return False
            
            ptr_injected_height.SetValue(INJECTED_IMAGE_HEIGHT if is_enabled else ptr_injected_height.GetMax())

    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False
    
    return True

# This function acquires and saves 10 images from a device; please see
# Acquisition example for more in-depth comments on acquiring images.
def acquire_images(cam, nodemap, nodemap_tldevice):
    result = True
    print('\n*** IMAGE ACQUISITION ***')

    try:
        # Set acquisition mode to continuous
        ptr_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if not PySpin.IsWritable(ptr_acquisition_mode):
            print('Unable to set acquisition mode to continuous (node retrieval). Aborting...')
            return False
        
        ptr_acquisition_mode_continuous = PySpin.CEnumEntryPtr(ptr_acquisition_mode.GetEntryByName("Continuous"))
        if not PySpin.IsReadable(ptr_acquisition_mode_continuous):
            print("'Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting...")
            return False
        
        acquisition_mode_continuous = ptr_acquisition_mode_continuous.GetValue()

        ptr_acquisition_mode.SetIntValue(int(acquisition_mode_continuous))

        # Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        ptr_string_serial = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
        if PySpin.IsReadable(ptr_string_serial):
            device_serial_number = ptr_string_serial.GetValue()
            print('Device serial number retrieved as %s' %device_serial_number)
        print('\n')

        # Retrieve, convert, and save images
        num_images = 10

        for i in range(num_images):
            try:
                result_image = cam.GetNextImage(1000)

                if result_image.IsIncomplete():
                    print('Image incomplete with image status %d ...' % result_image.GetImageStatus())
                else:
                    print('Grabbed Image %d, width = %d, height = %d' \
                        % (i, result_image.GetWidth(), result_image.GetHeight()))
                    
                    result = display_chunk_data(result_image)
                
                # Release image
                result_image.Release()
                print('')
            
            except PySpin.SpinnakerException as ex:
                print('Unexpected exception: %s' % ex)
                result = False

        cam.EndAcquisition()
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
        return False

    return result

# This function acts as the body of the example; please see NodeMapInfo example
# for more in-depth comments on setting up cameras.
def run_single_camera(cam):
    result = False
    err = 0

    try:
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        result = print_device_info(nodemap_tldevice)

        cam.Init()

        nodemap = cam.GetNodeMap()

        # Check to make sure camera supports inference 
        print('Checking camera inference support...')
        ptr_inference_enable = PySpin.CBooleanPtr(nodemap.GetNode('InferenceEnable'))
        if not PySpin.IsWritable(ptr_inference_enable):
            print('Inference is not supported on this camera. Aborting...')
            return False

        # Upload custom inference network onto the camera
        # The inference network file is in a movidius specific neural network format.
        # Uploading the network to the camera allows for "inference on the edge" where
        # camera can apply deep learning on a live stream. Refer to "Getting Started
        # with Firefly-DL" for information on how to create your own custom inference
        # network files using pre-existing neural network.
        err = upload_file_to_camera(nodemap, "InferenceNetwork", NETWORK_FILE_PATH)
        if err != True:
            return err

        # Upload injected test image
        # Instead of applying deep learning on a live stream, the camera can be
        # tested with an injected test image.        
        err = upload_file_to_camera(nodemap, "InjectedImage", INJECTED_IMAGE_FILE_PATH)
        if err != True:
            return err

        # Configure inference
        err = configure_inference(nodemap, True)
        if err != True:
            return err

        # Configure test pattern to make use of the injected image
        err = configure_test_pattern(nodemap, True)
        if err != True:
            return err

        # Configure trigger
        # When enabling inference results via chunk data, the results that accompany a frame
        # will likely not be the frame that inference was run on. In order to guarantee that
        # the chunk inference results always correspond to the frame that they are sent with,
        # the camera needs to be put into the "inference sync" trigger mode.
        # Note: Enabling this setting will limit frame rate so that every frame contains new
        #       inference dataset. To not limit the frame rate, you can enable InferenceFrameID
        #       chunk data to help determine which frame is associated with a particular
        #       inference data.
        err = configure_trigger(nodemap)
        if err != True:
            return err

        # Configure chunk data
        err = configure_chunk_data(nodemap)
        if err != True:
            return err

        # Acquire images and display chunk data
        result = result | acquire_images(cam, nodemap, nodemap_tldevice)

        # Disable chunk data
        err = disable_chunk_data(nodemap)
        if err != True:
            return err

        # Disable trigger
        err = disable_trigger(nodemap)
        if err != True:
            return err

        # Disable test pattern
        err = configure_test_pattern(nodemap, False)
        if err != True:
            return err

        # Disable inference
        err = configure_inference(nodemap, False)
        if err != True:
            return err

        # Clear injected test image
        err = delete_file_on_camera(nodemap, "InjectedImage")
        if err != True:
            return err

        # Clear uploaded inference network
        err = delete_file_on_camera(nodemap, "InferenceNetwork")
        if err != True:
            return err

        # Deinitialize camera
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Unexpected exception: %s' % ex)
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

    result = False

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))    
    
    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %s\n' % num_cameras)

    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False
    
    for i, cam in enumerate(cam_list):
        print('Running example for camera %d...' % i)
        result = result | run_single_camera(cam)
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