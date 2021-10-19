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
# FileAccess_QuickSpin.py shows shows how to read and write images using camera File Access function.
#   
# This example uploads an image to the camera File Access storage and also
# downloads the image from the camera File Access storage and saves it to
# the disk.
#
# It also provides debug message when an additional argument `--verbose` is passed in,  
# giving more detailed status of the progress to the users. 
# 
# Run with arguments in format (no quotes):  "--mode </d or /u>  --verbose (optional)"
#           /d: Download saved image from camera and save it to the working directory.
#           /u: Grab an image and store it on camera.
#

import PySpin
import numpy as np
import os
import argparse
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
subparsers = parser.add_subparsers()

class ImageAcquisitionUtil:
    @staticmethod
    def check_node_readable(node):
        return PySpin.IsAvailable(node) and PySpin.IsReadable(node)

    @staticmethod
    def grab_reference_image(cam):
        """
        This function first grabs 5 images to stablize the camera,
        then it grabs a reference image and returns its pointer.
        
        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: Pointer to the reference image
        :rtype: ImagePtr
        """
        reference_image = PySpin.Image.Create()

        # Start capturing images
        cam.BeginAcquisition()

        # Grab a couple of images to stabilize the camera
        for image_count in range(5):
            try:
                result_image = cam.GetNextImage(1000)
                if result_image.IsIncomplete():
                    print('Imgae incomplete with image status %s' % result_image.GetImageStatus())
                else:
                    print('Grabbed image %s' %str(image_count) + ', width = %s' % str(result_image.GetWidth())\
                                                               + ', height = %s' % str(result_image.GetHeight()))
                    reference_image.DeepCopy(result_image)
                result_image.Release()
            except PySpin.SpinnakerException as ex:
                print(ex)
                continue

        cam.EndAcquisition()

        return reference_image

class FileAccess:
    @staticmethod
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

            if ImageAcquisitionUtil.check_node_readable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(),
                                    node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
                print('')
            else:
                print('Device control information not available.')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    @staticmethod
    def execute_delete_command(cam):
        """
        This function executes delete operation on the camera.

        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            cam.FileOperationSelector.SetValue(PySpin.FileOperationSelector_Delete)
            cam.FileOperationExecute.Execute()

            if cam.FileOperationStatus.GetValue() != PySpin.FileOperationStatus_Success:
                print('Failed to delete file!')
                return False
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return True

    @staticmethod
    def open_file_to_write(cam):
        """
        This function opens the camera file for writing.

        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            cam.FileOperationSelector.SetValue(PySpin.FileOperationSelector_Open)
            cam.FileOpenMode.SetValue(PySpin.FileOpenMode_Write)
            cam.FileOperationExecute.Execute()

            if cam.FileOperationStatus.GetValue() != PySpin.FileOperationStatus_Success:
                print('Failed to open file for writing!')
                return False
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return True

    @staticmethod
    def execute_write_command(cam):
        """
        This function executes write command on the camera.

        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            cam.FileOperationSelector.SetValue(PySpin.FileOperationSelector_Write)
            cam.FileOperationExecute.Execute()

            if cam.FileOperationStatus.GetValue() != PySpin.FileOperationStatus_Success:
                print('Failed to write to file!')
                return False
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception : %s' % ex)
            return False
        return True

    @staticmethod
    def close_file(cam):
        """
        This function closes the file. 

        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            cam.FileOperationSelector.SetValue(PySpin.FileOperationSelector_Close)
            cam.FileOperationExecute.Execute()

            if cam.FileOperationStatus.GetValue() != PySpin.FileOperationStatus_Success:
                print('Failed to close file!')
                return False
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return True

    @staticmethod
    def upload_image(cam, verbose=False):
        """
        This function first acquires a reference image from the camera, 
        then it writes the image file to the camera with file selector UserFile1.

        :param cam: Camera used to download file from.
        :param verbose: Prints additional details of file download (False by default)
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            success = True

            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            FileAccess.print_device_info(nodemap_tldevice)
            
            cam.Init()

            # Check file selector support
            print('Checking file selector support')
            if cam.FileSelector.GetAccessMode() == PySpin.NA or cam.FileSelector.GetAccessMode() == PySpin.NI:
                print('File selector not supported on device!')
                return False

            # Apply small pixel format
            if ImageAcquisitionUtil.check_node_readable(cam.PixelFormat.GetEntry(PySpin.PixelFormat_Mono8)):
                cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
            else:
                # Use Bayer8 if Mono8 is not available
                cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerGB8)

            # Display camera setup information
            print('Width: %s' % cam.Width.GetValue())
            print('Height: %s' % cam.Height.GetValue())
            print('offsetX: %s' % cam.OffsetX.GetValue())
            print('OffsetY: %s' % cam.OffsetY.GetValue())
            print('PixelFormat: %s' % cam.PixelFormat.GetValue())

            # Grab reference image
            try:
                reference_image = ImageAcquisitionUtil.grab_reference_image(cam)
            except PySpin.SpinnakerException as ex:
                cam.DeInit()
                del cam
                print('Unexpected error grabbing reference image: %s' % ex)
                return False

            # Form file path
            filename = "DeviceStreamWrite-"
            if cam.DeviceSerialNumber.GetAccessMode() == PySpin.RW or cam.DeviceSerialNumber.GetAccessMode() == PySpin.RO:
                filename += "%s-" % cam.DeviceSerialNumber.ToString()
            filename += ".bmp"

            # Save image
            reference_image.Save(filename)
            print('Image saved at %s' % filename)

            print('*** UPLOADING IMAGE ***')

            # Perform file stream write
            selector_list = cam.FileSelector.GetEntries()

            for entry in selector_list:
                # Get current enum entry node
                node = PySpin.CEnumEntryPtr(entry)
                
                if verbose:
                    print('\nChecking FileSelector EnumEntry - %s' % node.GetSymbolic())

                # Check file selector entry support
                if not node or not ImageAcquisitionUtil.check_node_readable(node):
                    # Go to next entry node
                    print('%s not supported!' % node.GetSymbolic())
                    continue

                if node.GetSymbolic() == "UserFile1":
                    # Set file selector
                    cam.FileSelector.SetIntValue(int(node.GetNumericValue()))

                    # Delete file on camera before writing in case camera runs out of space
                    file_size = cam.FileSize.GetValue()
                    if file_size > 0:
                        if not FileAccess.execute_delete_command(cam):
                            print('Failed to delete file!')
                            success = False
                            continue

                    # Open file on camera for write
                    if not FileAccess.open_file_to_write(cam):
                        print('Failed to open file!')
                        success = False
                        continue

                    # Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
                    if cam.FileAccessLength.GetValue() < cam.FileAccessBuffer.GetLength():
                        try:
                            cam.FileAccessLength.SetValue(cam.FileAccessBuffer.GetLength())
                        except PySpin.SpinnakerException as ex:
                            print('Unable to set FileAccessLength to FileAccessBuffer length: %s' % ex)

                    # Set file access offset to zero if it's not
                    cam.FileAccessOffset.SetValue(0)

                    # Compute number of write operations required
                    total_bytes_to_write = reference_image.GetBufferSize()
                    intermediate_buffer_size = cam.FileAccessLength.GetValue()
                    write_iterations = (total_bytes_to_write // intermediate_buffer_size) + \
                                    (0 if ((total_bytes_to_write % intermediate_buffer_size) == 0) else 1)
                        
                    if total_bytes_to_write == 0:
                        print('Empty Image. No data will be written to camera.')
                        return False

                    if verbose:
                        print('')
                        print('Total bytes to write: %s' % total_bytes_to_write)
                        print('FileAccessLength: %s' % intermediate_buffer_size)
                        print('Write iterations: %s' % write_iterations)

                    bytes_left_to_write = total_bytes_to_write
                    total_bytes_written = 0

                    print('Writing data to device')

                    # Splitting the file into equal chunks (except the last chunk)
                    sections = []
                    for index in range(write_iterations):
                        offset = index * intermediate_buffer_size
                        if offset == 0:
                            continue
                        sections.append(offset)
                    
                    # Get image data and split into equal chunks
                    image_data = reference_image.GetData()
                    split_data = np.array_split(image_data, sections)

                    for i in range(len(split_data)):
                        # Setup data to write
                        tmp_buffer = split_data[i]

                        # Write to AccessBufferNode
                        cam.FileAccessBuffer.Set(tmp_buffer)

                        if intermediate_buffer_size > bytes_left_to_write:
                            # Update FileAccessLength, otherwise garbage data outside the range would be written to device
                            cam.FileAccessLength.SetValue(bytes_left_to_write)

                        # Perform write command
                        if not FileAccess.execute_write_command(cam):
                            print('Writing stream failed!')
                            success = False
                            break

                        # Verify size of bytes written
                        size_written = cam.FileOperationResult.GetValue()

                        # Log current file access offset
                        if verbose:
                            print('File Access Offset: %s' % cam.FileAccessOffset.GetValue())

                        # Keep track of total bytes written
                        total_bytes_written += size_written
                        if verbose:
                            print('Bytes written: %s of %s' % (total_bytes_written, total_bytes_to_write))

                        # Keep track of bytes left to write
                        bytes_left_to_write = total_bytes_to_write - total_bytes_written
                        
                        if verbose:
                            print('Progress: (%s//%s)' % (i, write_iterations))
                        else:
                            print('Progress: %s' % int((i*100 / write_iterations)) + "%")

                    print('Writing complete')

                    if not FileAccess.close_file(cam):
                        success = False

            cam.DeInit()
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return success

    @staticmethod
    def open_file_to_read(cam):
        """
        This function opens the file to read.

        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            cam.FileOperationSelector.SetValue(PySpin.FileOperationSelector_Open)
            cam.FileOpenMode.SetValue(PySpin.FileOpenMode_Read)
            cam.FileOperationExecute.Execute()

            if cam.FileOperationStatus.GetValue() != PySpin.FileOperationStatus_Success:
                print('Failed to open file for reading!')
                return False
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return True

    @staticmethod
    def execute_read_command(cam):
        """
        This function executes read command on the camera.

        :param cam: Camera used to perform file operation.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            cam.FileOperationSelector.SetValue(PySpin.FileOperationSelector_Read)
            cam.FileOperationExecute.Execute()

            if cam.FileOperationStatus.GetValue() != PySpin.FileOperationStatus_Success:
                print('Failed to read file!')
                return False
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return True

    @staticmethod
    def download_image(cam, verbose=False):
        """
        This function reads the image file stored in the camera file selector UserFile1,
        saving the file to the working directory of this example. 

        :param cam: Camera used to download file from.
        :param verbose: Prints additional details of file download (False by default)
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            success = True

            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            FileAccess.print_device_info(nodemap_tldevice)

            cam.Init()

            # Check file selector support
            print('Checking file selector support')
            if cam.FileSelector.GetAccessMode() == PySpin.NA or cam.FileSelector.GetAccessMode() == PySpin.NI:
                print('File selector not supported on device!')
                return False

            print('*** DOWNLOADING IMAGE ***')

            selector_list = cam.FileSelector.GetEntries()

            for entry in selector_list:
                node = PySpin.CEnumEntryPtr(entry)
                if verbose:
                    print('\nChecking FileSelector EnumEntry - %s' % node.GetSymbolic())

                # Check file selector entry support
                if not node or not ImageAcquisitionUtil.check_node_readable(node):
                    # Go to next entry node
                    print('%s not supported!' % node.GetSymbolic())
                    continue
                
                # Use UserFile1 as the selector in this example.
                # Available file selector entries varies across different cameras
                if node.GetSymbolic() == "UserFile1":
                    # Set file selector
                    cam.FileSelector.SetIntValue(int(node.GetNumericValue()))

                    # Get file size
                    total_bytes_to_read = cam.FileSize.GetValue()
                    if total_bytes_to_read == 0:
                        print('%s - No data available to read!' % node.GetSymbolic())
                        success = False
                        continue

                    print('Total data to download: %s' % total_bytes_to_read)

                    # Open file on camera for reading
                    if not FileAccess.open_file_to_read(cam):
                        print('Failed to open file!')
                        success = False
                        continue

                    # Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
                    if cam.FileAccessLength.GetValue() < cam.FileAccessBuffer.GetLength():
                            try:
                                cam.FileAccessLength.SetValue(cam.FileAccessBuffer.GetLength())
                            except PySpin.SpinnakerException as ex:
                                print('Unable to set FileAccessLength to FileAccessBuffer length: %s' % ex)

                    # Set file access offset to zero
                    cam.FileAccessOffset.SetValue(0)

                    # Computer number of read operations required
                    intermediate_buffer_size = cam.FileAccessLength.GetValue()
                    read_iterations = (total_bytes_to_read // intermediate_buffer_size) + \
                                (0 if ((total_bytes_to_read % intermediate_buffer_size) == 0) else 1)

                    if verbose:
                        print('')
                        print('Total bytes to read: %s' % total_bytes_to_read)
                        print('FileAccessLength: %s' % intermediate_buffer_size)
                        print('Write iterations: %s' % read_iterations)

                    print('Fetching image from camera.')

                    total_size_read = 0
                    size_read = cam.FileOperationResult.GetValue()
                    image_data = np.array(size_read, dtype=np.uint8)

                    for i in range(read_iterations):
                        if not FileAccess.execute_read_command(cam):
                            print('Reading stream failed!')
                            success = False
                            break

                        # Verify size of bytes read
                        size_read = cam.FileOperationResult.GetValue()

                        # Read from buffer Node
                        buffer_read = cam.FileAccessBuffer.Get(size_read)
                        if i == 0:
                            image_data = buffer_read
                        else:
                            image_data = np.append(image_data, buffer_read)

                        # Keep track of total bytes read
                        total_size_read += size_read
                        if verbose:
                            print('Bytes read: %s of %s' % (total_size_read, total_bytes_to_read))
                            print('Progress: (%s//%s)' % (i, read_iterations))
                        else:
                            print('Progress: %s' % int((i*100 / read_iterations)) + "%")

                    print('Reading complete')

                    if not FileAccess.close_file(cam):
                        success = False

                    # Form file path
                    filename = "DeviceStreamRead-"

                    if cam.DeviceSerialNumber.GetAccessMode() == PySpin.RW or cam.DeviceSerialNumber.GetAccessMode() == PySpin.RO:
                        filename += "%s-" % cam.DeviceSerialNumber.ToString()

                    filename += ".bmp"

                    # Image should be captured with Mono8 or Bayer8, it sets camera to correct pixel format
                    # in order to grab image ROI
                    if ImageAcquisitionUtil.check_node_readable(cam.PixelFormat.GetEntry(PySpin.PixelFormat_Mono8)):
                        cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
                    elif ImageAcquisitionUtil.check_node_readable(cam.PixelFormat.GetEntry(PySpin.PixelFormat_BayerGB8)):
                        # Use Bayer8 if Mono8 is not available
                        cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerGB8)
                    else:
                        print('Failed to set camera pixel format.')
                        return False

                    width = cam.Width.GetValue()
                    height = cam.Height.GetValue()
                    offset_x = cam.OffsetX.GetValue()
                    offset_y = cam.OffsetY.GetValue()
                    pixel_format = cam.PixelFormat.GetValue()

                    # Form image and save data
                    print('Width: %s' % width)
                    print('Height: %s' % height)
                    print('OffsetX: %s' % offset_x)
                    print('OffsetY: %s' % offset_y)
                    print('PixelFormat: %s' % pixel_format)

                    # Create image
                    image = PySpin.Image.Create(width, height, offset_x, offset_y, pixel_format, image_data)

                    # Save image
                    image.Save(filename)
                    print('Image saved at %s' % filename)

            cam.DeInit()
        except PySpin.SpinnakerException as ex:
            print('Unexpected exception: %s' % ex)
            return False
        return success

def main():
    """
    Example entry point; please see Enumeration.py example for more in-depth
    comments on preparing and cleaning up the system with PySpin.

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
    
    parser = argparse.ArgumentParser()
    parser = subparsers.add_parser('stop', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--mode', required=True, type=str,
                        help='/u : Grab an image and store it on camera.\n/d : Download saved image from camera and save it to the working directory.\n')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Enable verbose output.')

    args = parser.parse_args()

    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()

    # This example only works with 1 camera is connected. 
    if num_cameras == 0:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False
    elif num_cameras > 1:
        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('This example only works when 1 camera is connected.')
        input('Done! Press Enter to exit...')
        return False
    else:
        if args.mode == '/u' or args.mode == '/U':
            result = FileAccess.upload_image(cam_list[0], args.verbose)
        elif args.mode == '/d' or args.mode == '/D':
             result = FileAccess.download_image(cam_list[0], args.verbose)
        else:
            print("Invalid Argument! Use '--help' to learn available arguments.")
            input('Done! Press Enter to exit...')
            return False

    if not result:
        print('File Access failed')
    else:
        print('File Access is successful!')

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