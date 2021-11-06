###############################################################################
# FLIR balckfly video capture
# Uses PySpin to capture images
# Uses OpenCV to resize and roate images
# Urs Utzinger 
#
# Fall 2020, First release
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock

#
import logging
import time
import sys

# Open FLIR driver
import PySpin
from PySpin.PySpin import TriggerMode_Off, TriggerSource_Software

# Open Computer Vision
import cv2

###############################################################################
# Video Capture
################################################################################

class blackflyCapture(Thread):
    """
    This thread continually captures frames from a blackfly camera
    """
    #
    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    ############################################################################
    def __init__(self, configs, camera_num: int = 0, res: (int, int) = None, exposure: float = None):
        # initialize 
        self.logger     = logging.getLogger("backflyCapture{}".format(camera_num))

        # populate desired settings from configuration file or function arguments
        ########################################################################
        self._camera_num     = camera_num
        if exposure is not None:
            self._exposure    = exposure  
        else: 
            self._exposure   = configs['exposure']
        if res is not None:
            self._camera_res = res
        else: 
            self._camera_res = configs['camera_res']
        self._output_res     = configs['output_res']
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._flip_method    = configs['flip']
        self._framerate      = configs['fps']
        self._autoexposure   = configs['autoexposure']       # autoexposure depends on camera
        self._binning        = configs['binning']
        self._framerate      = configs['fps']
        self._offset         = configs['offset']
        self._adc            = configs['adc']
        self._trigout        = configs['trigout']            # -1 no trigout, 1 = line 1
        self._ttlinv         = configs['ttlinv']             # False = normal, True=inverted
        self._trigin         = configs['trigin']             # -1 no trigin,  1 = line 1

        # Threading Locks, Events
        self.frame_lock      = Lock()
        self.stopped         = True

        # Open up the Camera
        self._open_capture()

        # Init Frame and Thread
        self.frame        = None
        self.new_frame    = False
        self.frame_time   = 0.0
        self.measured_fps = 0.0

        Thread.__init__(self)

    #
    # Setup the Camera
    ############################################################################
    def _open_capture(self):
        """Open up the camera so we can begin capturing frames"""
        #
        # Open Library and Camera
        #########################
        self.system = PySpin.System.GetInstance()
        # Get current library version
        self.version = self.system.GetLibraryVersion()
        self.logger.log(logging.INFO, "Status:Driver:Version: {}.{}.{}.{}".format(self.version.major,  self.version.minor, self.version.type, self.version.build))
        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        self.num_cameras = self.cam_list.GetSize()
        self.logger.log(logging.INFO, "Status:Number of Cameras: {}".format(self.num_cameras))
        # Finish if there are no cameras
        if self.num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()
            # Release system instance
            self.system.ReleaseInstance()
            self.logger.log(logging.CRITICAL, "Status: No cameras found!")
            self.capture_open = False
            self.cam = None
            return False
        # Open the camera
        self.cam = self.cam_list[self._camera_num]
        # Get device information
        self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        self.node_device_information = PySpin.CCategoryPtr(self.nodemap_tldevice.GetNode('DeviceInformation'))
        if PySpin.IsAvailable(self.node_device_information) and PySpin.IsReadable(self.node_device_information):
            features = self.node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                if PySpin.IsReadable(node_feature): 
                    self.logger.log(logging.INFO, "Status:Cameras Features: {} {}".format(node_feature.GetName(), node_feature.ToString()))
                else: 
                    self.logger.log(logging.WARNING, "Status:Cameras Features: {}".format('Node not readable'))
        else:
            self.logger.log(logging.WARNING, "Status:Cameras Features: {}".format('Device control information not available.'))

        #
        # Initialize camera
        ###################
        self.cam.Init()
        self.capture_open = True

        #
        # Camera Settings
        #################

        # 1 Turn off features
        #   ISP (off)
        #   Automatic Gain (off)
        #   Gamma (set to 1.0 then off)
        #   Automatic Exposure (off preferred for high frame rate)
        # Some more needed off for version 2.5
        #   Acquisition Mode = Continous
        #   Acquisiton Frame Rate Enable = True
        #   Exposure Mode Timed
        #   Gamme Enable Off
        
        # ISP OFF
        if self.cam.IspEnable.GetAccessMode() == PySpin.RW:
            self.cam.IspEnable.SetValue(False)
            self.logger.log(logging.INFO, "Status:Camera:ISP Enable: {}".format(self.cam.IspEnable.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:ISP Enable: no access")
        # Gain OFF
        if self.cam.GainSelector.GetAccessMode() == PySpin.RW: 
            self.cam.GainSelector.SetValue(PySpin.GainSelector_All)
            self.logger.log(logging.INFO, "Status:Camera:GainSelector: {}".format(self.cam.GainSelector.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:GainSelector: no access")
        if self.cam.Gain.GetAccessMode() == PySpin.RW:
            self.cam.Gain.SetValue(1.0)
            self.logger.log(logging.INFO, "Status:Camera:Gain: {}".format(self.cam.Gain.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:Gain: no access")
        if self.cam.GainAuto.GetAccessMode() == PySpin.RW:
            self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            self.logger.log(logging.INFO, "Status:Camera:GainAuto: {}".format(self.cam.GainAuto.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:GainAuto: no access")
        #   Acquisition Mode = Continous
        if self.cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
            self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.logger.log(logging.INFO, "Status:Camera:AcquistionMode: {}".format(self.cam.AcquisitionMode.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:AcquisionMode: no access")
        #   Exposure Mode Timed
        if self.cam.ExposureMode.GetAccessMode() == PySpin.RW:
            self.cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)        
            self.logger.log(logging.INFO, "Status:Camera:ExposureMode: {}".format(self.cam.ExposureMode.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:ExposureMode: no access")
        #   Acquisiton Frame Rate Enable = True
        if self.cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
            self.cam.AcquisitionFrameRateEnable.SetValue(True)
            self.logger.log(logging.INFO, "Status:Camera:AcquisionFrameRateEnable: {}".format(self.cam.AcquisitionFrameRateEnable.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:AcquisionFrameRateEnable: no access")
        #   Gamma Off
        if self.cam.GammaEnable.GetAccessMode() == PySpin.RW:
            self.cam.GammaEnable.SetValue(True)
        if self.cam.Gamma.GetAccessMode() == PySpin.RW:
            self.cam.Gamma.SetValue(1.0)
            self.logger.log(logging.INFO, "Status:Camera:Gamma: {}".format(self.cam.Gamma.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:Gamma: no access")
        if self.cam.GammaEnable.GetAccessMode() == PySpin.RW:
            self.cam.GammaEnable.SetValue(False)
            self.logger.log(logging.INFO, "Status:Camera:GammaEnable: {}".format(self.cam.GammaEnable.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:GammaEnable: no access")
        # features changable by client
        self.autoexposure = self._autoexposure # using user accessible function
        #
        # 2 Set Sensor
        #   Binning, should be set before setting width and height
        #   Width and Height
        #   Offset, should be set after setting binning and resolution
        #   Bit Depth (8, 10, 12 or 14bit), will also set Pixel Format to either Mono8 or Mono16, affects frame rte
        # hard coded features
        if self.cam.BinningVerticalMode.GetAccessMode() == PySpin.RW:
            self.cam.BinningVerticalMode.SetValue(PySpin.BinningVerticalMode_Sum)
            self.logger.log(logging.INFO, "Status:Camera:BinningVerticalMode: {}".format(self.cam.BinningVerticalMode.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:BinningVerticalMode: no access")
        if self.cam.BinningHorizontalMode.GetAccessMode() == PySpin.RW:
            self.cam.BinningHorizontalMode.SetValue(PySpin.BinningHorizontalMode_Sum)
            self.logger.log(logging.INFO, "Status:Camera:BinningHorizonalMode: {}".format(self.cam.BinningHorizontalMode.GetValue()))
        else:
            self.logger.log(logging.WARNING, "Status:Camera:BinningHorizonalMode: no access")
        # features changeable by user
        self.binning    = self._binning
        self.resolution = self._camera_res
        self.offset     = self._offset
        self.adc        = self._adc

        #
        # 3a Digital Input for Trigger
        #   Set Input Trigger, if set to -1 use software trigger
        self.trigin  = self._trigin

        #
        # 3 Digital Output
        #   Set Output Trigger, Line 1 has opto isolator but transitions slow, line 2 takes about 4-10 us to transition
        self.trigout = self._trigout 

        #
        # 4 Aquistion
        #   Continous 
        #   FPS, should be set after turning off auto feature, ADC bit depth, binning as they slow down camera
        # hard coded features
        # feaures changable by client
        self.exposure = self._exposure
        self.fps = self._framerate

        #
        # Start Image Acquistion
        ########################  
        self.cam.BeginAcquisition() # Start Aquision
        # if trigger source is Software: execute, otherwise nothing goes
        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
        if self.cam.TriggerSource.GetValue() == PySpin.TriggerSource_Software:
            self.cam.TriggerSoftware()
            self.logger.log(logging.INFO, "Status:Camera:TriggerSource: executed")
        else:
            self.logger.log(logging.WARNING, "Status:Camera:TriggerSource: no access")

        self.logger.log(logging.INFO, "Status:Acquiring images.")

    #
    # Thread routines
    # Start Stop and Update Thread
    #####################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        time.sleep(0.1)                # give thread time to finish
        self.cam.EndAcquisition()
        self.cam.DeInit()
        # self.cam.Release()
        del self.cam
        self.cam_list.Clear()          # clear camera list before releasing system
        self.system.ReleaseInstance()  # release system instance

    def start(self, capture_queue = None):
        """ set the thread start conditions """
        self.stopped = False
        T = Thread(target=self.update, args=(capture_queue,))
        T.daemon = True # run in background
        T.start()

    def update(self, capture_queue):   # this runs until thread is stopped
        """ run the thread """
        last_fps_time = time.time()
        num_frames = 0
        while not self.stopped:
            current_time = time.time()

            # Get New Image
            if self.cam is not None:
                image_result = self.cam.GetNextImage(1000) # timeout in ms, function blocks until timeout
                if not image_result.IsIncomplete(): # should always be complete
                    # self.frame_time = self.cam.EventExposureEndTimestamp.GetValue()
                    img = image_result.GetNDArray() # get inmage as NumPy array
                    image_result.Release() # make next frame available
                    num_frames += 1
                    self.frame_time = int(current_time*1000)

                    # avoid unnecessary memory copy and changing of array sizes
                    if (self._output_height <= 0) and (self._flip_method == 0):
                        if capture_queue is not None:
                            if not capture_queue.full():
                                capture_queue.put((self.frame_time, img), block=False)
                            else:
                                self.logger.log(logging.WARNING, "Status:Capture Queue is full!")
                        else:
                            self.frame = img
                    else:
                        # adjust output height
                        tmp = cv2.resize(img, self._output_res)
                        # flip resized image
                        if   self._flip_method == 0: # no flipping
                            tmpf = tmp
                        elif self._flip_method == 1: # ccw 90
                            tmpf = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                        elif self._flip_method == 2: # rot 180, same as flip lr & up
                            tmpf = cv2.roate(tmp, cv.ROTATE_180)
                        elif self._flip_method == 3: # cw 90
                            tmpf = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                        elif self._flip_method == 4: # horizontal
                            tmpf = cv2.flip(tmp, 0)
                        elif self._flip_method == 5: # upright diagonal. ccw & lr
                            tmp_tmp = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                            tmpf = cv2.flip(tmp_tmp, 1)
                        elif self._flip_method == 6: # vertical
                            tmpf = cv2.flip(tmp, 1)
                        elif self._flip_method == 7: # upperleft diagonal
                            tmpf = cv2.transpose(tmp)
                        else:
                            tmpf = tmp
                        if capture_queue is not None:
                            if not capture_queue.full():
                                capture_queue.put((self.frame_time, tmpf), block=False)
                            else:
                                self.logger.log(logging.WARNING, "Status:Capture Queue is full!")                                    
                        else:
                            self.frame = tmpf

            # FPS calculation
            if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.logger.log(logging.INFO, "Status:FPS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time

    #
    # Frame routines
    # If queue is not used
    #########################################################################

    @property
    def frame(self):
        """ returns most recent frame """
        with self.frame_lock:
            self._new_frame = False
        return self._frame
    @frame.setter
    def frame(self, val):
        """ set new frame content """
        with self.frame_lock:
            self._frame = val
            self._new_frame = True

    @property
    def new_frame(self):
        """ check if new frame available """
        with self.frame_lock:
            return self._new_frame
    @new_frame.setter
    def new_frame(self, val):
        """ override wether new frame is available """
        with self.frame_lock:
            self._new_frame = val

    #
    # Camera routines
    # Reading and setting camera options
    ###################################################################

    @property
    def width(self):
        """ returns video capture width """
        if self.capture_open:
            self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())  
            return self._camera_res[0]
        else: return -1
    @width.setter
    def width(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            self.logger.log(logging.WARNING, "Status:Camera:Width not changed:{}".format(val))
            return
        if self.capture_open:
            val = max(self.cam.Width.GetMin(), min(self.cam.Width.GetMax(), val))
            if self.cam.Width.GetAccessMode() == PySpin.RW:
                self.cam.Width.SetValue(val)
                self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())
                self.logger.log(logging.INFO, "Status:Camera:Width:{}".format(val))
            else:
                self.logger.log(logging.ERROR, "Status:Camera:Failed to set width to {}!".format(val))
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set resolution, camera not open!")

    @property
    def height(self):
        """ returns video capture height """
        if self.capture_open:
            self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())  
            return self._camera_res[1]
        else: return -1
    @height.setter
    def height(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            self.logger.log(logging.WARNING, "Status:Camera:Height not changed:{}".format(val))
            return
        if self.capture_open:
            val = max(self.cam.Height.GetMin(), min(self.cam.Height.GetMax(), val)) 
            if self.cam.Height.GetAccessMode() == PySpin.RW:
                self.cam.Height.SetValue(val)
                self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())
                self.logger.log(logging.INFO, "Status:Camera:Height:{}".format(val))
            else:
                self.logger.log(logging.ERROR, "Status:Camera:Failed to set height to {}!".format(val))
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set resolution, camera not open!")

    @property
    def resolution(self):
        """ returns current resolution width x height """
        if self.capture_open:
            self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())
            return self._camera_res
        else: return (-1, -1)
    @resolution.setter
    def resolution(self, val):
        """ sets video capture resolution """
        if val is None: return
        if self.capture_open:
            if len(val) > 1: # we have width x height
                _tmp0 = max(self.cam.Width.GetMin(),  min(self.cam.Width.GetMax(),  val[0]))
                _tmp1 = max(self.cam.Height.GetMin(), min(self.cam.Height.GetMax(), val[1]))
                val = (_tmp0, _tmp1)
                if self.cam.Width.GetAccessMode() == PySpin.RW:
                    self.cam.Width.SetValue(int(val[0]))
                    self.logger.log(logging.INFO, "Status:Camera:Width:{}".format(val[0]))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set width to {}!".format(val[0]))
                if self.cam.Height.GetAccessMode() == PySpin.RW:
                    self.cam.Height.SetValue(int(val[1]))
                    self.logger.log(logging.INFO, "Status:Camera:Height:{}".format(val[1]))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set height to {}!".format(val[1]))
                self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())
            else: # given only one value for resolution, make image square
                val = max(self.cam.Width.GetMin(), min(self.cam.Width.GetMax(), val))
                val = max(self.cam.Height.GetMin(), min(self.cam.Height.GetMax(), val))
                if self.cam.Width.GetAccessMode() == PySpin.RW:
                    self.cam.Width.SetValue(int(val))
                    self.logger.log(logging.INFO, "Status:Camera:Width:{}".format(val))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set resolution to {},{}!".format(val,val))
                if self.cam.Height.GetAccessMode() == PySpin.RW:
                    self.cam.Height.SetValue(int(val)) 
                    self.logger.log(logging.INFO, "Status:Height:{}".format(val))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set resolution to {},{}!".format(val,val))
                self._camera_res = (self.cam.Width.GetValue(), self.cam.Height.GetValue())
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set resolution, camera not open!")

    @property
    def offset(self):
        """ returns current sesor offset """
        if self.capture_open:
            self._offset = (self.cam.OffsetX.GetValue(), self.cam.OffsetY.GetValue())
            return self._offset
        else: return (float("NaN"), float("NaN")) 
    @resolution.setter
    def offset(self, val):
        """ sets sensor offset """
        if val is None: return
        if self.capture_open:
            if len(val) > 1: # have horizontal and vertical
                _tmp0 = max(min(self.cam.OffsetX.GetMin(), val[0]), self.cam.OffsetX.GetMax())
                _tmp1 = max(min(self.cam.OffsetY.GetMin(), val[1]), self.cam.OffsetY.GetMax())
                val = (_tmp0, _tmp1)
                if self.cam.OffsetX.GetAccessMode() == PySpin.RW:
                    self.cam.OffsetX.SetValue(int(val[0]))
                    self.logger.log(logging.INFO, "Status:Camera:OffsetX:{}".format(val[0]))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set X offset to {}!".format(val[0]))
                if self.cam.OffsetY.GetAccessMode() == PySpin.RW:
                    self.cam.OffsetY.SetValue(int(val[1]))
                    self.logger.log(logging.INFO, "Status:Camera:OffsetY:{}".format(val[1]))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Y offset to {}!".format(val[1]))
                self._offset = (self.cam.OffsetX.GetValue(), self.cam.OffsetY.GetValue())
            else: # given only one value for resolution
                val = max(min(self.cam.OffsetX.GetMin(), val), self.cam.OffsetX.GetMax())
                val = max(min(self.cam.OffsetY.GetMin(), val), self.cam.OffsetY.GetMax())
                if self.cam.OffsetX.GetAccessMode() == PySpin.RW:
                    self.cam.OffsetX.SetValue(int(val))
                    self.logger.log(logging.INFO, "Status:Camera:OffsetX:{}".format(val))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set X offset to {}!".format(val))
                if self.cam.OffsetY.GetAccessMode() == PySpin.RW:
                    self.cam.OffsetY.SetValue(int(val))
                    self.logger.log(logging.INFO, "Status:Camera:OffsetY:{}".format(val))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Y offset to {},{}!".format(val))
                self._offset = (self.cam.OffsetX.GetValue(), self.cam.OffsetY.GetValue())
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set offset, camera not open!")

    @property
    def binning(self):
        """ returns binning horizontal, vertical """
        if self.capture_open:
            self._binning = (self.cam.BinningHorizontal.GetValue(), self.cam.BinningVertical.GetValue())
            return self._binning
        else: return (-1, -1)
    @resolution.setter
    def binning(self, val):
        """ sets sensor biginning """
        if val is None: return
        if self.capture_open:
            if len(val) > 1: # have horizontal x vertical
                _tmp0 = min(max(val[0], self.cam.BinningHorizontal.GetMin()), self.cam.BinningHorizontal.GetMax()) 
                _tmp1 = min(max(val[1], self.cam.BinningVertical.GetMin()), self.cam.BinningVertical.GetMax())
                val = (_tmp0, _tmp1)
                if self.cam.BinningHorizontal.GetAccessMode() == PySpin.RW:
                    self.cam.BinningHorizontal.SetValue(int(val[0]))
                    self.logger.log(logging.INFO, "Status:Camera:BinningHorizontal:{}".format(val[0]))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set horizontal binning to {}!".format(val[0]))
                if self.cam.BinningVertical.GetAccessMode() == PySpin.RW:
                    self.cam.BinningVertical.SetValue(int(val[1]))
                    self.logger.log(logging.INFO, "Status:Camera:BinningVertical:{}".format(val[1]))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set vertical binning to {}!".format(val[1]))
                self._binning = (self.cam.BinningHorizontal.GetValue(), self.cam.BinningVertical.GetValue())
            else: # given only one value for resolution
                _tmp0 = min(max(val[0], self.cam.BinningHorizontal.GetMin()), self.cam.BinningHorizontal.GetMax()) 
                _tmp1 = min(max(val[1], self.cam.BinningVertical.GetMin()), self.cam.BinningVertical.GetMax()) 
                val = (_tmp0, _tmp1)
                if self.cam.BinningHorizontal.GetAccessMode() == PySpin.RW:
                    self.cam.BinningHorizontal.SetValue(int(val))
                    self.logger.log(logging.INFO, "Status:Camera:BinningHorizontal:{}".format(val))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set horizontal binning to {}!".format(val))
                if self.cam.BinningVertical.GetAccessMode() == PySpin.RW:
                    self.cam.BinningVertical.SetValue(int(val))
                    self.logger.log(logging.INFO, "Status:Camera:BinningVertical:{}".format(val))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set vertical binning to {}!".format(val))
                self._binning = (self.cam.BinningHorizontal.GetValue(), self.cam.BinningVertical.GetValue())
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set binning, camera not open!")

    @property
    def exposure(self):
        """ returns curent exposure """
        if self.capture_open:
            self._exposure = capture.ExposureTime.GetValue() 
            return self._exposure
        else: return float("NaN")
    @exposure.setter
    def exposure(self, val):
        """ sets exposure """
        if (val is None) or (val == -1):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set exposure to {}!".format(val))
            return
        # Setting exposure implies that autoexposure is off and exposure mode is timed 
        if self.cam.ExposureMode.GetValue() != PySpin.ExposureMode_Timed:
            self.logger.log(logging.ERROR, "Status:Camera:Can not set exposure! Exposure Mode needs to be Timed")
            return
        if self.cam.ExposureAuto.GetValue() != PySpin.ExposureAuto_Off:
            self.logger.log(logging.ERROR, "Status:Camera:Can not set exposure! Exposure is Auto")
            return
        # Setting exposure
        if self.capture_open:
            self._exposure = max(self.cam.ExposureTime.GetMin(), min(self.cam.ExposureTime.GetMax(), float(val)))
            if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
                self.cam.ExposureTime.SetValue(self._exposure)
                self.logger.log(logging.INFO, "Status:Camera:Exposure:{}".format(self._exposure))
            else:
                self.logger.log(logging.ERROR, "Status:Camera:Failed to set expsosure to:{}".format(self._exposure))
            self._exposure = self.cam.ExposureTime.GetValue()
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set exposure, camera not open!")

    @property
    def autoexposure(self):
        """ returns curent auto exposure state """
        if self.capture_open:
            if (self.cam.ExposureAuto.GetValue() == PySpin.ExposureAuto_Continuous) or (self.cam.ExposureAuto.GetValue() == PySpin.ExposureAuto_Once):
                self._autoexposure = 1
            else:
                self._autoexposure = 0
            return self._autoexposure
        else: return -1
    @autoexposure.setter
    def autoexposure(self, val):
        """ sets autoexposure """
        # On:
        # 1) Turn on autoexposure
        # 2) Update FPS as autoexposure reduces framerate
        # Off:
        # 1) Turn off autoexposre
        # 2) Set exposure 
        # 3) Set max FPS
        if (val is None) or (val == -1):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set Autoexposure to:{}".format(val))
            return
        if self.capture_open:
            if val > 0: 
                # Setting Autoexposure on
                if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
                    self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
                    self.logger.log(logging.INFO, "Status:Camera:Autoexposure:{}".format(1))
                    self._autoexposure = 1
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Autoexposure to:{}".format(val))
                self._framerate = self.cam.AcquistionFrameRate.GetMax()
                if self.cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                    self.cam.AcquisitionFrameRate.SetValue(self._framerate)
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Frame Rate to:{}".format(self._framerate))
            else:
                # Setting Autoexposure off
                if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
                    self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                    self.logger.log(logging.INFO, "Status:Camera:Autoexposure:{}".format(0))
                    self._autoexposure = 0
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Autoexposure to:{}".format(val))
                self._exposure = max(self.cam.ExposureTime.GetMin(), min(self.cam.ExposureTime.GetMax(), self._exposure))
                if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
                    self.cam.ExposureTime.SetValue(self._exposure)
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Exposure Time to:{}".format(self._exposure))
                self._framerate = self.cam.AcquisitionFrameRate.GetMax()
                if self.cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                    self.cam.AcquisitionFrameRate.SetValue(self._framerate)
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set Frame Rate to:{}".format(self._framerate))
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set auto exposure, camera not open!")

    @property
    def fps(self):
        """ returns current frames per second setting """
        if self.capture_open:
            self._framerate = self.cam.AcquisitionFrameRate.GetValue() 
            return self._framerate
        else: return float("NaN")
    @fps.setter
    def fps(self, val):
        """ set frames per second in camera """
        if (val is None) or (val == -1):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set framerate to:{}".format(val))
            return
        if self.capture_open:
            self._framerate = float(val)
            if self.cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                self.cam.AcquisitionFrameRate.SetValue(self._framerate)
                self.logger.log(logging.INFO, "Status:Camera:FPS:{}".format(self._framerate))
            else:
                self.logger.log(logging.ERROR, "Status:Camera:Failed to set FPS to:{}".format(self._framerate))
            self._framerate = self.cam.AcquisitionFrameRate.GetValue()

    @property
    def adc(self):
        """ returns adc bitdetpth """
        if self.capture_open:
            _tmp = self.cam.AdcBitDepth.GetValue()
            if _tmp == PySpin.AdcBitDepth_Bit8:
                self._adc = 8
            elif _tmp == PySpin.AdcBitDepth_Bit10:
                self._adc = 10
            elif _tmp == PySpin.AdcBitDepth_Bit12:
                self._adc = 12
            elif _tmp == PySpin.AdcBitDepth_Bit14:
                self._adc = 14
            else:
                self._adc = -1
            return self._adc
        else: return -1
    @adc.setter
    def adc(self, val):
        """ sets adc bit depth """
        if (val is None) or (val == -1):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set exposure to {}!".format(val))
            return
        if self.capture_open:
            if val == 8:
                if self.cam.AdcBitDepth.GetAccessMode() == PySpin.RW:
                    self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit8) 
                    self._adc = 8
                    self.logger.log(logging.INFO, "Status:Camera:ADC:{}".format(self._adc))
                    if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
                        self.logger.log(logging.INFO, "Status:Camera:PixelFormat:{}".format('Mono8'))
                    else:
                        self.logger.log(logging.ERROR, "Status:Camera:Failed to set Pixel Format to:{}".format('Mono8'))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set ADC to:{}".format(val))
            elif val == 10:
                if self.cam.AdcBitDepth.GetAccessMode() == PySpin.RW:
                    self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10) 
                    self._adc = 10
                    self.logger.log(logging.INFO, "Status:Camera:ADC:{}".format(self._adc))
                    if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
                        self.logger.log(logging.INFO, "Status:Camera:PixelFormat:{}".format('Mono16'))
                    else:
                        self.logger.log(logging.ERROR, "Status:Camera:Failed to set Pixel Format to:{}".format('Mono16'))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set ADC to:{}".format(val))
            elif val == 12:
                if self.cam.AdcBitDepth.GetAccessMode() == PySpin.RW:
                    self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit12) 
                    self._adc = 12
                    self.logger.log(logging.INFO, "Status:Camera:ADC:{}".format(self._adc))
                    if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
                        self.logger.log(logging.INFO, "Status:Camera:PixelFormat:{}".format('Mono16'))
                    else:
                        self.logger.log(logging.ERROR, "Status:Camera:Failed to set Pixel Format to:{}".format('Mono16'))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set ADC to:{}".format(val))
            elif val == 14:
                if self.cam.AdcBitDepth.GetAccessMode() == PySpin.RW:
                    self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit14) 
                    self._adc = 14
                    self.logger.log(logging.INFO, "Status:Camera:ADC:{}".format(self._adc))
                    if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
                        self.logger.log(logging.INFO, "Status:Camera:PixelFormat:{}".format('Mono16'))
                    else:
                        self.logger.log(logging.ERROR, "Status:Camera:Failed to set Pixel Format to:{}".format('Mono16'))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set ADC to:{}".format(val))
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set ADC, camera not open!")

    @property
    def ttlinv(self):
        """ returns tigger output ttl polarity """
        if self.capture_open:
            return self._ttlinv
        else: return -1
    @ttlinv.setter
    def ttlinv(self, val):
        """ sets trigger logic polarity """
        if (val is None):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set trigger level to:{}!".format(val))
            return
        if self.capture_open:
            if val == 0: # Want regular trigger output polarity
                self.cam.LineInverter.SetValue(False)
                self._ttlinv = False
            elif val == 1: # want inverted trigger output polarity
                self.cam.LineInverter.SetValue(True)
                self._ttlinv = True

            self.logger.log(logging.INFO, "Status:Camera:Trigger Output Logic Inverted:{}".format(self._ttlinv))
        else: # camera not open
            self.logger.log(logging.INFO, "Status:Camera:Failed to set Trigger Output Polarity, camera not open!")

    @property
    def trigout(self):
        """ returns tigger output setting """
        if self.capture_open:
            return self._trigout
        else: return -1
    @trigout.setter
    def trigout(self, val):
        """ sets trigger output line """

        # Line Selector Line 0,1,2,3
        # Line Mode Out
        #   0 Input Only
        #   1 Output Only
        #   2 Input and Output
        #   3 Input Only
        # Line Inverer True or False
        # Line Source Exposure Active

        if (val is None):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set trigger output on line:{}!".format(val))
            return
        if self.capture_open:
            if val == -1: # dont want trigger output
                # set line 2 to input
                if self.cam.LineSelector.GetAccessMode() == PySpin.RW:
                    self.cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
                if self.cam.LineMode.GetAccessMode() == PySpin.RW:
                    self.cam.LineMode.SetValue(PySpin.LineMode_Input)
                if self.cam.LineSource.GetAccessMode() == PySpin.RW:
                    self.cam.LineSource.SetValue(PySpin.LineSource_Off) 
                # set line 1 to output (only option)
                if self.cam.LineSelector.GetAccessMode() == PySpin.RW:
                    self.cam.LineSelector.SetValue(PySpin.LineSelector_Line1) 
                if self.cam.LineMode.GetAccessMode() == PySpin.RW:
                    self.cam.LineMode.SetValue(PySpin.LineMode_Output) # dont have input
                if self.cam.LineInverter.GetAccessMode() == PySpin.RW:
                    self.cam.LineInverter.SetValue(self._ttlinv)
                if self.cam.LineSource.GetAccessMode() == PySpin.RW:
                    self.cam.LineSource.SetValue(PySpin.LineSource_ExposureActive) # dont have off
                self._trigout = -1
                self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
            elif val == 1: # want trigger output on line 1, need pullup to 3V on line 1, set line 2 to 3V
                # set line 2 to out high for 3V
                if self.cam.LineSelector.GetAccessMode() == PySpin.RW:
                    self.cam.LineSelector.SetValue(PySpin.LineSelector_Line2) 
                if self.cam.LineMode.GetAccessMode() == PySpin.RW:
                    self.cam.LineMode.SetValue(PySpin.LineMode_Output)
                if self.cam.LineInverter.GetAccessMode() == PySpin.RW:
                    self.cam.LineInverter.SetValue(False)
                if self.cam.LineSource.GetAccessMode() == PySpin.RW:
                    self.cam.LineSource.SetValue(PySpin.LineSource_Off)
                # set line 1 to Exposure Active
                if self.cam.LineSelector.GetAccessMode() == PySpin.RW:
                    self.cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
                if self.cam.LineMode.GetAccessMode() == PySpin.RW:
                    self.cam.LineMode.SetValue(PySpin.LineMode_Output)
                if self.cam.LineInverter.GetAccessMode() == PySpin.RW:
                    self.cam.LineInverter.SetValue(self._ttlinv)
                if self.cam.LineSource.GetAccessMode() == PySpin.RW:
                    self.cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)
                self._trigout = 1
                self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
            elif val == 2: # best option
                # Line Selector Line 2
                # Line Mode Out
                # Line Inverer True or False
                # Line Source Exposure Active
                if self.cam.LineSelector.GetAccessMode() == PySpin.RW:
                    self.cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
                if self.cam.LineMode.GetAccessMode() == PySpin.RW:
                    self.cam.LineMode.SetValue(PySpin.LineMode_Output)
                if self.cam.LineInverter.GetAccessMode() == PySpin.RW:
                    self.cam.LineInverter.SetValue(self._ttlinv)
                if self.cam.LineSource.GetAccessMode() == PySpin.RW:
                    self.cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)
                self._trigout = 2
                self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set trigger, camera not open!")

    @property
    def trigin(self):
        """ returns tigger input setting """
        if self.capture_open:
            return self._trigin
        else: return -1
    @trigin.setter
    def trigin(self, val):
        """ sets trigger input line """

        if (val is None):
            self.logger.log(logging.ERROR, "Status:Camera:Can not set trigger input on line {}!".format(val))
            return
        if self.capture_open:
            if val == -1: # no external trigger, trigger source is software
                if self.cam.TriggerSelector.GetAccessMode() == PySpin.RW:
                    self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_AcquisitionStart)
                    if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                    if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                    if self.cam.TriggerOverlap.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
                    if self.cam.TriggerDelay.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerDelay.SetValue(self.cam.TriggerDelay.GetMin())
                    self._trigout = -1
                    self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set trigger output on line:{}".format(self._trigout))

            elif val == 0: # trigger is line 0
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                    self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                    if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                        if self.cam.TriggerActivation.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                        if self.cam.TriggerOverlap.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
                        if self.cam.TriggerDelay.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerDelay.SetValue(self.cam.TriggerDelay.GetMin())
                        self._trigout = 0
                        self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set trigger output on line:{}".format(self._trigout))
            elif val == 2: # trigger is line 2
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                    self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                    if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line2)
                        if self.cam.TriggerActivation.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                        if self.cam.TriggerOverlap.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
                        if self.cam.TriggerDelay.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerDelay.SetValue(self.cam.TriggerDelay.GetMin())
                        self._trigout = 2
                        self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set trigger output on line:{}".format(self._trigout))
            elif val == 3: # trigger is line 3
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                    self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                    if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
                        if self.cam.TriggerActivation.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                        if self.cam.TriggerOverlap.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
                        if self.cam.TriggerDelay.GetAccessMode() == PySpin.RW:
                            self.cam.TriggerDelay.SetValue(self.cam.TriggerDelay.GetMin())
                        self._trigout = 3
                        self.logger.log(logging.INFO, "Status:Camera:Trigger Output:{}".format(self._trigout))
                else:
                    self.logger.log(logging.ERROR, "Status:Camera:Failed to set trigger output on line:{}".format(self._trigout))
        else: # camera not open
            self.logger.log(logging.ERROR, "Status:Camera:Failed to set trigger, camera not open!")

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    from camera.configs.blackfly_configs  import configs

    logging.basicConfig(level=logging.DEBUG)
 
    print("Starting Capture")
    camera = blackflyCapture(camera_num=0, res=(720,540))
    camera.start()

    print("Getting Frames")
    window_handle = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    while(cv2.getWindowProperty("Camera", 0) >= 0):
        if camera.new_frame:
            cv2.imshow('Camera', camera.frame)
            # print(camera.frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.stop()

####################################################################
#
# Dump of BFS-U3-04S2M System Settings
#
####################################################################

#
# Root
#     Device Information
#         Device ID: USB\VID_1E10&PID_4000&MI_00\6&214B0...
#         Device Serial Number: 20397374
#         Device Vendor Name  : FLIR
#         Device Model Name: Blackfly S BFS-U3-04S2M
#         Device Type: USB3Vision
#         Device Display Name: FLIR Blackfly S BFS-U3-04S2M
#         Device Access Status: OpenReadWrite
#         Device Version: 1707.1.6.0
#         Device Driver Version: PGRUSBCam3.sys : 2.7.3.249
#         Device User ID:
#         Device Is Updater Mode: False
#         DeviceInstanceId: USB\VID_1E10&PID_4000&MI_00\6&214B0...
#         Device Location: 0000.0014.0000.017.000.000.000.000....
#         Device Current Speed: SuperSpeed
#         GUI XML Source: Device
#         GUI XML Path: Input.xml
#         GenICam XML Source: Device
#         GenICam XML Path: 
#         Device Is In U3V Protocol: True

#     Device Control
#         Device Endianess Mechanism: Standard


# *** PRINTING TL STREAM NODEMAP ***

# Root
#     Stream Information
#         Stream ID  : 0
#         Stream Type: USB3Vision

#     Buffer Handling Control
#         Manual Stream Buffer Count: 10
#         Resulting Stream Buffer Count: 10
#         Stream Buffer Count Max: 2767
#         Stream Buffer Count Mode: Manual
#         Stream Buffer Handling Mode: OldestFirst
#         Stream Announce Buffer Minimum: 1
#         Stream Announced Buffer Count: 0
#         Stream Started Frame Count: 0
#         Stream Delivered Frame Count: 0
#         Stream Lost Frame Count: 0
#         Stream Input Buffer Count: 0
#         Stream Output Buffer Count: 0
#         Stream Block Transfer Size: 1048576
#         CRC Check Enable: False
#         Stream Is Grabbing: False
#         Stream Chunk Count Maximum: 50
#         Stream Buffer Alignment: 1024

#     Stream Diagnostics
#         Failed Buffer Count: 0


# *** PRINTING GENICAM NODEMAP ***

# Root
#     Acquisition Control
#         Acquisition Mode: Continuous[*], SingleFrame, MultiFrame
#         Acquisition Frame Count: 2
#         Acquisition Burst Frame Count: 1
#         Exposure Mode: Timed[*], Trigger Width
#         Exposure Time: 998.0
#         Exposure Auto: Continuous, Off[*], Once
#         Acquisition Frame Rate: 199.8653706863057
#         Resulting Frame Rate: 199.86529059413954
#         Acquisition Frame Rate Enable: False, True to enable changes
#         Acquisition Line Rate: 255427.84137931035, Not sure where
#         Trigger Selector: FrameStart[*], Aquistion Start, Frame Burst Start
#         Trigger Mode: Off[*], On
#         Trigger Source: Software[*], Line 0..3, User Output 0..2, Counter.., Logic..
#         Trigger Overlap: Off, Read Out
#         Trigger Delay: 9.0
#         Sensor Shutter Mode: Global

#     Analog Control
#         Gain Selector: All
#         Gain: 0.0
#         Gain Auto: Continuous, Off[*], Once
#         Black Level Selector: All, Analog, Digital
#         Black Level: 0.0
#         Black Level Clamping Enable: True
#         Gamma: 0.800048828125
#         Gamma Enable: True, False

#     Image Format Control
#         Sensor Width: 720[*]
#         Sensor Height: 540[*]
#         Width Max: 720
#         Height Max: 540
#         Width: 720[*]
#         Height: 540[*]
#         Offset X: 0[*]
#         Offset Y: 0[*]
#         Pixel Format: Mono8[*], Mono16, Mono10Packed, Mono12Packer, Mono10p, Mono12p
#         Pixel Size: Bpp8
#         Pixel Color Filter: None
#         Pixel Dynamic Range Min: 0
#         Pixel Dynamic Range Max: 255
#         ISP Enable: False
#         Binning Selector: All[*], ISP, Sensor
#         Binning Horizontal Mode: Sum[*], Average
#         Binning Vertical Mode: Sum[*], Average
#         Binning Horizontal: 1[*],2,4
#         Binning Vertical: 1[*],2,4
#         Decimation Selector: All[*], Sensor
#         Decimation Horizontal Mode: Discard[*]
#         Decimation Vertical Mode: Discard[*]
#         Decimation Horizontal: 1[*]
#         Decimation Vertical: 1[*]
#         Reverse X: False[*]
#         Reverse Y: False[*]
#         Test Pattern Generator Selector: Sensor
#         Test Pattern: Off
#         ADC Bit Depth: Bit10,  Bit8, Bit12

#     Device Control
#         Device Scan Type: Areascan
#         Device Vendor Name: FLIR
#         Device Model Name: Blackfly S BFS-U3-04S2M
#         Sensor Description: Sony IMX287 (1/2.9" Mono CMOS)
#         Device Manufacturer Info: Aug 31 2017 11:10:27
#         Device Version: 1707.1.6.0
#         Device Firmware Version: 1707.1.6.0
#         Device Serial Number: 20397374
#         Device ID: 20397374
#         Device User ID:
#         Device TL Type: USB3Vision
#         Device Gen CP Version Major: 1
#         Device Gen CP Version Minor: 0
#         Device Max Throughput: 77777600
#         Device Link Speed: 500000000
#         Device Link Throughput Limit: 380000000, 500000000[*]
#         Device Link Bandwidth Reserve: 0.0
#         Device Link Current Throughput: 77725213
#         Device Indicator Mode: Active
#         Device Temperature Selector: Sensor
#         Device Temperature: 39.625
#         Timestamp Latch Value: 0
#         Timestamp Increment: 1000
#         Device Power Supply Selector: External
#         Power Supply Voltage: 4.8740234375
#         Power Supply Current: 0.441162109375
#         Device Uptime: 103
#         Link Uptime: 95
#         Enumeration Count: 1
#         Max Device Reset Time: 30000

#     Transport Layer Control
#         Payload Size: 388800
#         TLParamsLocked: 0
#         USB3 Vision
#             Max Response Time: 200
#             Message Channel: 0
#             Access Privilege: 0
#             U3V Version Major: 1
#             U3V Version Minor: 0
#             U3V Capability: 0
#             U3V SIRM Available: True
#             U3V EIRM Available: True
#             U3V IIDC2 Available: False
#             Max Command Transfer Length: 1024
#             Max Ack Transfer Length: 1024
#             Number of Stream Channels: 1
#             Current Speed: SuperSpeed

#         Link Error Count: 0
#         Link Recovery Count: 0

#     Sequencer Control
#         Sequencer Mode: Off[*], On
#         Sequencer Configuration Mode: Off[*], On
#         Sequencer Configuration Valid: No
#         Sequencer Configuration Reset: Resets the sequencer configuration ...
#         Sequencer Feature Selector: ExposureTime, Gain
#         Sequencer Feature Enable: True, False
#         Sequencer Set Start: 0
#         Sequencer Set Selector: 0
#         Sequencer Set Valid: No
#         Sequencer Set Save: Saves the current device configurat...
#         Sequencer Set Load: Loads currently selected sequencer ...
#         Sequencer Path Selector: 0
#         Sequencer Trigger Source: Off, Start Frame

#     Color Transformation Control

#     Auto Algorithm Control
#         ROI Selector: Awb
#         Target Grey Value Auto: Continuous
#         Lighting Mode: Normal
#         Metering Mode: Average
#         Exposure Time Lower Limit: 100.0
#         Exposure Time Upper Limit: 15000.0
#         Gain Lower Limit: 0.0
#         Gain Upper Limit: 18.000065071923338
#         Target Grey Value Lower Limit: 3.9100684261974585
#         Target Grey Value Upper Limit: 93.841642228739
#         EV Compensation: 0.0
#         Auto Exposure Damping: 0.5
#         Auto Exposure Control Priority: Gain

#     Flat Field Correction Control
#         Flat Field User Table Control


#     Defective Pixel Correction
#         Defect Correct Static Enable: True
#         Defect Correction Mode: Average
#         Defect Table Pixel Count: 1
#         Defect Table Index: 0
#         Defect X Coordinate: 704
#         Defect Y Coordinate: 352
#         Defect Table Apply: Applies the current defect table, s...
#         Defect Table Factory Restore: Restores the factory default eeprom...

#     User Set Control
#         User Set Selector: Default
#         User Set Load: Loads the User Set specified by Use...
#         User Set Save: Saves the User Set specified by Use...
#         User Set Default: Default
#         User Set Feature Selector: AasRoiEnableAe
#         User Set Feature Enable: True

#     Chunk Data Control
#         Chunk Mode Active: False
#         Chunk Selector: FrameID
#         Chunk Enable: False
#         Chunk Gain Selector: All
#         Chunk Black Level Selector: All

#     LUT Control
#         LUT Selector: LUT1
#         LUT Enable: False
#         LUT Index: 0
#         LUT Value: 0

#     Event Control
#         Event Selector: Error
#         Event Notification: Off
#         Event Exposure End Data
#             Event Exposure End: 40003

#         Event Error Data
#             Event Error: 40000

#         Event Serial Port Receive Data
#             Event Serial Port Receive: 40005

#         Event Test Data
#             Event Test: 20479


#     Counter And Timer Control
#         Counter Selector: Counter0
#         Counter Event Source: MHzTick
#         Counter Duration: 1
#         Counter Value: 1
#         Counter Trigger Source: ExposureStart
#         Counter Trigger Activation: RisingEdge, Level Low, Level High, Falling Edge, Any Edge
#         Counter Status: CounterTriggerWait
#         Counter Delay: 0

#     Test Control
#         Test Pending Ack: 0
#         Test 0001: 0
#         GUI XML Manifest Address: 4026535968

#     Logic Block Control
#         Logic Block Selector: LogicBlock0
#         Logic Block LUT Selector: Value
#         Logic Block LUT Input Selector: Input0
#         Logic Block LUT Input Source: Zero
#         Logic Block LUT Input Activation: LevelHigh
#         Logic Block LUT Output Value All: 255
#         Logic Block LUT Row Index: 0
#         Logic Block LUT Output Value: True

#     Digital IO Control
#         Line Selector: Line0..3, Line 1[*]
#         Line Mode: Input, Output[*]
#         Line Inverter: False
#         Line Status: False (read)
#         Line Status All: 12
#         Input filter Selector: Deglitch
#         Line Filter Width: 0.0
#         Line Source: Off
#         Line Format: OptoCoupled (read)
#         User Output Selector: UserOutput0..3
#         User Output Value: False, True`
#         User Output Value All: 0

#     Serial Port Control (Not in SpinView)
#         Serial Port Selector: SerialPort0
#         Serial Port Source: Off
#         Serial Port Baud Rate: Baud57600
#         Serial Port Data Bits: 8
#         Serial Port Stop Bits: Bits1
#         Serial Port Parity: None
#         Transmit Queue Max Character Count: 4096
#         Transmit Queue Current Character Count: 0
#         Receive Queue Max Character Count: 4096
#         Receive Queue Current Character Count: 0
#         Receive Framing Error Count: 0
#         Receive Parity Error Count: 0

#     File Access
#         File Selector: UserSetDefault
#         File Operation Selector: Open
#         File Open Mode: Read
#         File Access Offset: 0
#         File Access Length: 1
#         File Operation Status: Success
#         File Operation Result: 0
#         File Size: 1376

#     Transfer Control
#         Transfer Control Mode: Basic
#         Transfer Queue Max Block Count: 130
#         Transfer Queue Current Block Count: 0
#         Transfer Queue Overflow Count: 0