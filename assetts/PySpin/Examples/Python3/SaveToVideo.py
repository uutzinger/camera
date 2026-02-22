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
# SaveToVideo.py shows how to create a video from a vector of
# images. It relies on information provided in the Enumeration, Acquisition,
# and NodeMapInfo examples.
#
# This example introduces the SpinVideo class, which is used to quickly and
# easily create various types of AVI/MP4 videos. It demonstrates the creation of
# four types: uncompressed, MJPG, H264 (AVI) and H264 (MP4).
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import PySpin
import sys

NUM_IMAGES = 100
class VideoType:
    """'Enum' to select video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264_AVI = 2
    H264_MP4 = 3

chosenVideoType = VideoType.UNCOMPRESSED

# The duration of a video stream depends on the numbe´r of encoded images
# and the stream frame rate:
#
#     VIDEO_DURATION (second) = NUM_IMAGES / STREAM_FRAME_RATE
#
# eg. NUM_IMAGES = 100 and STREAM_FRAME_RATE = 20fps will result in a
# 5 seconds video stream.
#
# For normal playback speed, STREAM_FRAME_RATE should be set to the
# acquisition frame rate (useCustomFrameRate = false).
# However, the STREAM_FRAME_RATE can be customized (useCustomFrameRate = true,
# and adjusting customFrameRate)
use_custom_framerate = False
custom_framerate = 1.0

def save_list_to_video(nodemap, nodemap_tldevice, images):
    """
    This function prepares, saves, and cleans up a video from a vector of images.

    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param images: List of images to save to an AVI video.
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type images: list of ImagePtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** CREATING VIDEO ***')

    try:
        result = True

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        if PySpin.IsReadable(node_serial):
            device_serial_number = node_serial.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Get the current frame rate; acquisition frame rate recorded in hertz
        #
        # *** NOTES ***
        # The video frame rate can be set to anything; however, in order to
        # have videos play in real-time, the acquisition frame rate can be
        # retrieved from the camera.

        node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))

        if not PySpin.IsReadable(node_acquisition_framerate):
            print('Unable to retrieve frame rate. Aborting...')
            return False

        framerate_to_set = node_acquisition_framerate.GetValue()
        if use_custom_framerate:
            framerate_to_set = custom_framerate

        print('Frame rate to be set to %d...' % framerate_to_set)

        # Select option and open filetype with unique filename
        #
        # *** NOTES ***
        # Depending on the filetype, a number of settings need to be set in
        # an object called an option. An uncompressed option only needs to
        # have the video frame rate set whereas videos with MJPG or H264
        # compressions should have more values set.
        #
        # Once the desired option object is configured, open the file
        # with the option in order to create the image file.
        #
        # Note that the filename does not need to be appended to the
        # name of the file. This is because the recorder object takes care
        # of the file extension automatically.
        #
        # *** LATER ***
        # Once all images have been added, it is important to close the file -
        # this is similar to many other standard file streams.

        video_recorder = PySpin.SpinVideo()

        if chosenVideoType == VideoType.UNCOMPRESSED:
            video_filename = 'SaveToVideo-Uncompressed-%s' % device_serial_number

            option = PySpin.AVIOption()
            option.frameRate = framerate_to_set
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        elif chosenVideoType == VideoType.MJPG:
            video_filename = 'SaveToVideo-MJPG-%s' % device_serial_number

            option = PySpin.MJPGOption()
            option.frameRate = framerate_to_set
            option.quality = 75
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        elif chosenVideoType == VideoType.H264_AVI or chosenVideoType == VideoType.H264_MP4:
            video_filename = 'SaveToVideo-H264-%s' % device_serial_number

            option = PySpin.H264Option()

            option.frameRate = framerate_to_set
            option.bitrate = 1000000
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()
            # Set this to true to save to a mp4 container
            option.useMP4 = (chosenVideoType == VideoType.H264_MP4)
            # Decrease this for a higher quality
            option.crf = 28

        else:
            print('Error: Unknown Type. Aborting...')
            return False

        video_recorder.Open(video_filename, option)

        # Construct and save video
        #
        # *** NOTES ***
        # Although the video file has been opened, images must be individually
        # appended in order to construct the video.
        print('Appending %d images to file: %s...' % (len(images), video_filename))

        for i in range(len(images)):
            video_recorder.Append(images[i])
            print('Appended image %d...' % i)

        # Close video file
        #
        # *** NOTES ***
        # Once all images have been appended, it is important to close the
        # Video file. Notice that once a video file has been closed, no more
        # images can be added.

        video_recorder.Close()
        print('Video saved at %s' % video_filename)

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
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('\n*** DEVICE INFORMATION ***\n')

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


def acquire_images(cam, nodemap):
    """
    This function acquires 30 images from a device, stores them in a list, and returns the list.
    please see the Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
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

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition mode set to continuous...')

        #  Begin acquiring images
        cam.BeginAcquisition()

        print('Acquiring images...')

        # Retrieve, convert, and save images
        images = list()

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
                    print('Image incomplete with image status %d...' % image_result.GetImageStatus())

                else:
                    #  Print image information; height and width recorded in pixels
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                    #  Convert image to mono 8 and append to list
                    images.append(processor.Convert(image_result, PySpin.PixelFormat_Mono8))

                    #  Release image
                    image_result.Release()
                    print('')

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                result = False

        # End acquisition
        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result, images


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run example on.
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

        # Acquire list of images
        err, images = acquire_images(cam, nodemap)
        if err < 0:
            return err

        result &= save_list_to_video(nodemap, nodemap_tldevice, images)

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

    print('Number of cameras detected:', num_cameras)

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
