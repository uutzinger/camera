###############################################################################
# Raspberry Pi CSI video capture
# Allows configuation of codec
# BitBuckets FRC 4183 & Urs Utzinger
# 2019, 2020
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread
from threading import Lock
from threading import Event

#
import logging
import time
import sys

# Open Computer Vision
import cv2
from   picamera import PiCamera
from   picamera.array import PiRGBArray

###############################################################################
# Video Capture
###############################################################################

class piCapture(Thread):
    """
    This thread continually captures frames from a CSI camera on Raspberry Pi
    """

    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    def __init__(self, configs, camera_num: int = 0, res: (int, int) = None, 
                 exposure: float = None):

        # initialize 
        self.logger     = logging.getLogger("piCapture{}".format(camera_num))

        # populate desired settings from configuration file or function call
        self.camera_num = camera_num
        if exposure is not None: self._exposure   = exposure
        else:                    self._exposure   = configs['exposure']
        if res is not None:      self._camera_res = res
        else:                    self._camera_res = (configs['camera_res'])
        self._capture_width                       = self._camera_res[0] 
        self._capture_height                      = self._camera_res[1]
        self._display_res                         = configs['output_res']
        self._display_width                       = self._display_res[0]
        self._display_height                      = self._display_res[1]
        self._framerate                           = configs['fps']
        self._flip_method                         = configs['flip']

        # Threading Locks, Events
        self.capture_lock    = Lock() # before changing capture settings lock them
        self.frame_lock      = Lock() # When copying frame, lock it
        self.stopped         = True

        # open up the camera
        self._open_capture()

        # Init Frame and Thread
        self.frame     = None
        self.new_frame = False
        self.stopped   = False
        self.measured_fps = 0.0

        Thread.__init__(self)

    def _open_capture(self):
        """Open up the camera so we can begin capturing frames"""
        # PiCamera
        self.capture      = PiCamera(self.camera_num)
        self.capture_open = not self.capture.closed

        # self.cv2SettingsDebug() # check camera properties

        if self.capture_open:
            # Apply settings to camera
            self.resolution = self._camera_res
            self.rawCapture = PiRGBArray(self.capture)
            self.fps          = self._framerate
            if self._exposure <= 0:
                # Auto Exposure and Auto White Balance
                ############################################################
                self.awb_mode     = 'auto'           # No auto white balance
                self.awb_gains    = (1,1)            # Gains for red and blue are 1
                self.brightness   = 50               # No change in brightness
                self.contrast     = 0                # No change in contrast
                self.drc_strength = 'off'            # Dynamic Range Compression off
                self.clock_mode   = 'raw'            # Frame numbers since opened camera
                self.color_effects= None             # No change in color
                self.fash_mode    = 'off'            # No flash
                self.image_denoise= False            # In vidoe mode
                self.image_effect = 'none'           # No image effects
                self.sharpness    = 0                # No changes in sharpness
                self.video_stabilization = False     # No image stablization
                self.exposure_mode= 'auto'             # automatic exposure control
                self.exposure_compensation = 0       # No automatic expsoure controls compensation
            else:
                self.exposure     = self._exposure                # camera exposure
                # Turn OFF Auto Features
                ############################################################
                self.awb_mode     = 'off'            # No auto white balance
                self.awb_gains    = (1,1)            # Gains for red and blue are 1
                self.brightness   = 50               # No change in brightness
                self.contrast     = 0                # No change in contrast
                self.drc_strength = 'off'            # Dynamic Range Compression off
                self.clock_mode   = 'raw'            # Frame numbers since opened camera
                self.color_effects= None             # No change in color
                self.fash_mode    = 'off'            # No flash
                self.image_denoise= False            # In vidoe mode
                self.image_effect = 'none'           # No image effects
                self.sharpness    = 0                # No changes in sharpness
                self.video_stabilization = False     # No image stablization
                self.iso          = 100              # Use ISO 100 setting, smallest analog and digital gains
                self.exposure_mode= 'off'            # No automatic exposure control
                self.exposure_compensation = 0       # No automatic expsoure controls compensation
            # Output Configuration
            self._display_res    = configs['output_res']
            self._display_width  = self._display_res[0]
            self._display_height = self._display_res[1]
        else:
            self.logger.log(logging.CRITICAL, "Status:Failed to open camera!")

    #
    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True

    def start(self):
        """ set the thread start conditions """
        self.stopped = False
        try:
            self.stream = self.capture.capture_continuous(self.rawCapture, 
                                                          format="bgr", use_video_port=True)
        except:
            self.logger.log(logging.CRITICAL, "Status:Failed to create camera stream!")

        T = Thread(target=self.update, args=())
        T.daemon = True # run in background
        T.start()


    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        last_fps_time = time.time()
        last_exposure_time = last_fps_time
        num_frames = 0
        for f in self.stream:
            img = f.array
            self.rawCapture.truncate(0)

            current_time = time.time()
            # FPS calculation
            if (current_time - last_fps_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                self.logger.log(logging.DEBUG, "Status:FPS:{}".format(self.measured_fps))
                num_frames = 0
                last_fps_time = current_time

            if self._display_height > 0:
                tmp = cv2.resize(img, self._display_res)
                if   self._flip_method == 0: # no flipping
                    self.frame = tmp
                elif self._flip_method == 1: # ccw 90
                    self.frame = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 2: # rot 180, same as flip lr & up
                    self.frame = cv2.roate(tmp, cv.ROTATE_180)
                elif self._flip_method == 3: # cw 90
                    self.frame = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 4: # horizontal
                    self.frame = cv2.flip(tmp, 0)
                elif self._flip_method == 5: # upright diagonal. ccw & lr
                    tmp = cv2.roate(tmp, cv.ROTATE_90_COUNTERCLOCKWISE)
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 6: # vertical
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 7: # upperleft diagonal
                    self.frame = cv2.transpose(tmp)
                else:
                    self.frame = tmp
            else:
                if   self._flip_method == 0: # no flipping
                    self.frame = img
                elif self._flip_method == 1: # ccw 90
                    self.frame = cv2.roate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 2: # rot 180, same as flip lr & up
                    self.frame = cv2.roate(img, cv.ROTATE_180)
                elif self._flip_method == 3: # cw 90
                    self.frame = cv2.roate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                elif self._flip_method == 4: # horizontal
                    self.frame = cv2.flip(img, 0)
                elif self._flip_method == 5: # upright diagonal. ccw & lr
                    tmp = cv2.roate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
                    self.frame = cv2.flip(tmp, 1)
                elif self._flip_method == 6: # vertical
                    self.frame = cv2.flip(img, 1)
                elif self._flip_method == 7: # upperleft diagonal
                    self.frame = cv2.transpose(img)
                else:
                    self.frame = img

            num_frames += 1


            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.capture.close()        

    #
    # Frame routines ##################################################
    # Each camera stores frames locally
    ###################################################################

    @property
    def frame(self):
        """ returns most recent frame """
        self._new_frame = False
        return self._frame

    @frame.setter
    def frame(self, img):
        """ set new frame content """
        with self.frame_lock:
            self._frame = img
            self._new_frame = True

    @property
    def new_frame(self):
        """ check if new frame available """
        out = self._new_frame
        return out

    @new_frame.setter
    def new_frame(self, val):
        """ override wether new frame is available """
        self._new_frame = val

    #
    # Pi CSI camera routines ##########################################
    # Rading and Setting Camera Options
    ###################################################################

    #
    # SONY IMX219
    # Mode	Resolution	Aspect	Framerate	    FoV	    Binning
    #                   Ratio
    # 1	    1920x1080	16:9	1/10 - 30	    Partial	None
    # 2	    3280x2464	4:3	    1/10 - 15	    Full	None
    # 3	    3280x2464	4:3	    1/10 - 15	    Full	None
    # 4	    1640x1232	4:3	    1/10 - 40	    Full	2x2
    # 5	    1640x922	16:9	1/10 - 40	    Full	2x2
    # 6	    1280x720	16:9	40 - 90 (120*)	Partial	2x2
    # 7	    640x480	    4:3	    40 - 90 (120*)	Partial	2x2

    # Omnivision OV5647
    # Mode	Resolution	Aspect 
    #                   Ratio	Framerate	    FoV	    Binning
    # 1	    1920x1080	16:9	1 - 30	        Partial	None
    # 2	    2592x1944	4:3	    1 - 15	        Full	None
    # 3	    2592x1944	4:3	    1/6 - 1	        Full	None
    # 4	    1296x972	4:3	    1 - 42	        Full	2x2
    # 5	    1296x730	16:9	1 - 49	        Full	2x2
    # 6	    640x480	    4:3	    42 - 60	        Full	4x4
    # 7	    640x480	    4:3	    60 - 90	        Full	4x4

    # write shutter_speed  sets exposure in microseconds
    # read  exposure_speed gives actual exposure 
    # shutter_speed = 0 then auto exposure
    # framerate determines maximum exposure

    # Read properties
    ###################################################################

    @property
    def resolution(self):            
        if self.capture_open: 
            return self.capture.resolution                            
        else: return float("NaN")

    @property
    def width(self):                 
        if self.capture_open: 
            return self.capture.resolution[0]
        else: return float("NaN")

    @property
    def height(self):                
        if self.capture_open: 
            return self.capture.resolution[1]                         
        else: return float("NaN")

    @property
    def fps(self):             
        if self.capture_open: 
            return self.capture.framerate+self.capture.framerate_delta 
        else: return float("NaN")

    @property
    def exposure(self):              
        if self.capture_open: 
            return self.capture.exposure_speed                        
        else: return float("NaN")

    # Write
    ###################################################################

    @resolution.setter
    def resolution(self, val):
        if val is None: return
        if self.capture_open:
            if len(val) > 1:
                with self.capture_lock: self.capture.resolution = val
                self.logger.log(logging.DEBUG, "Status:Width:{},Height:{}".format(val[0],val[1]))
                self._resolution = val
            else:
                with self.capture_lock: self.capture.resolution = (val, val)
                self.logger.log(logging.DEBUG, "Status:Width:{},Height:{}".format(val,val))
                self._resolution = (val, val)                    
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set resolution, camera not open!")

    @width.setter
    def width(self, val):
        if val is None: return
        val = int(val)
        if self.capture_open:
            with self.capture_lock: self.capture.resolution = (val, self.capture.resolution[1])
            self.logger.log(logging.DEBUG, "Status:Width:{}".format(val))

        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set resolution, camera not open!")

    @height.setter
    def height(self, val):
        if val is None: return
        val = int(val)
        if self.capture_open:
            with self.capture_lock: self.capture.resolution = (self.capture.resolution[0], val)
            self.logger.log(logging.DEBUG, "Status:Height:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set resolution, camera not open!")

    @fps.setter
    def fps(self, val):
        if val is None: return
        val = float(val)
        if self.capture_open:
            with self.capture_lock: self.capture.framerate = val
            self.logger.log(logging.DEBUG, "Status:FPS:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set franerate, camera not open!")

    @exposure.setter
    def exposure(self, val):
        if val is None: return
        if self.capture_open:
            with self.capture_lock: self.capture.shutter_speed  = val
            self.logger.log(logging.DEBUG, "Status:Exposure:{}".format(val))
            self._exposure = self.exposure
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set exposure, camera not open!")

    #
    # Color Balancing ##################################################
    #
    # Cannot set digital and analog gains, set ISO then read the gains.
    # awb_mode: can be off, auto, sunLight, cloudy, share, tungsten, fluorescent, flash, horizon, default is auto
    # analog gain: retreives the analog gain prior to digitization
    # digital gain: applied after conversion, a fraction
    # awb_gains: 0..8 for red,blue, typical values 0.9..1.9 if awb mode is set to "off:

    # Read
    @property
    def awb_mode(self):              
        if self.capture_open: 
            return self.capture.awb_mode               
        else: return float('NaN')

    @property
    def awb_gains(self):             
        if self.capture_open: 
            return self.capture.awb_gains              
        else: return float('NaN')

    @property
    def analog_gain(self):           
        if self.capture_open: 
            return self.capture.analog_gain           
        else: return float("NaN")

    @property
    def digital_gain(self):          
        if self.capture_open: 
            return self.capture.digital_gain           
        else: return float("NaN")

    # Write

    @awb_mode.setter
    def awb_mode(self, val):
        if val is None: return
        if self.capture_open:
            with self.capture_lock: self.capture.awb_mode  = val
            self.logger.log(logging.DEBUG, "Status:AWB Mode:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set autowb, camera not open!")
 
    @awb_gains.setter
    def awb_gains(self, val):
        if val is None: return
        if self.capture_open:
            if len(val) > 1:
                with self.capture_lock: self.capture.awb_gains  = val
                self.logger.log(logging.DEBUG, "Status:AWB Gains:red:{},blue:{}".format(val[0], val[1]))
            else:
                with self.capture_lock: self.capture.awb_gains = (val, val)
                self.logger.log(logging.DEBUG, "Status:AWB Gain:{},{}".format(val,val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set autowb gains, camera not open!")

    # Can not set analog and digital gains, needs special code
    #@analog_gain.setter
    #@digital_gain.setter

    #
    # Intensity and Contrast ###########################################
    #
    # brightness 0..100 default 50
    # contrast -100..100 default is 0
    # drc_strength is dynamic range compression strength; off, low, medium, high, default off
    # iso 0=auto, 100, 200, 320, 400, 500, 640, 800, on some cameras iso100 is gain of 1 and iso200 is gain for 2
    # exposure mode can be off, auto, night, nightpreview, backight, spotlight, sports, snow, beach, verylong, fixedfps, antishake, fireworks, default is auto, off fixes the analog and digital gains
    # exposure compensation -25..25, larger value gives brighter images, default is 0
    # meter_mode'average', 'spot', 'backlit', 'matrix'

    # Read
    @property
    def brightness(self):            
        if self.capture_open: 
            return self.capture.brightness             
        else: return float('NaN')

    @property
    def iso(self):                   
        if self.capture_open: 
            return self.capture.iso                    
        else: return float("NaN")

    @property
    def exposure_mode(self):         
        if self.capture_open: 
            return self.capture.exposure_mode          
        else: return float("NaN")

    @property
    def exposure_compensation(self): 
        if self.capture_open: 
            return self.capture.exposure_compensation  
        else: return float("NaN")

    @property
    def drc_strength(self):          
        if self.capture_open: 
            return self.capture.drc_strength           
        else: return float('NaN')

    @property
    def contrast(self):              
        if self.capture_open: 
            return self.capture.contrast               
        else: return float('NaN')

    # Write

    @brightness.setter
    def brightness(self, val):
        if val is None:  return
        val = int(val)
        if self.capture_open:
            with self.capture_lock: self.capture.brightness = val
            self.logger.log(logging.DEBUG, "Status:Brightness:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set brightnes, camera not open!")

    @iso.setter
    def iso(self, val):
        if val is None: return
        val = int(val)
        if self.capture_open:
            with self.capture_lock: self.capture.iso = val
            self.logger.log(logging.DEBUG, "Status:ISO:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set ISO, camera not open!")

    @exposure_mode.setter
    def exposure_mode(self, val):
        if val is None: return
        if self.capture_open:
            with self.capture_lock: self.capture.exposure_mode = val
            self.logger.log(logging.DEBUG, "Status:Exposure Mode:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set exposure mode, camera not open!")

    @exposure_compensation.setter
    def exposure_compensation(self, val):
        if val is None: return
        val = int(val)
        if self.capture_open:
            with self.capture_lock: self.capture.exposure_compensation = val
            self.logger.log(logging.DEBUG, "Status:Exposure Compensation:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set exposure compensation, camera not open!")

    @drc_strength.setter
    def drc_strength(self, val):
        if val is None: return
        if self.capture_open:
            with self.capture_lock: self.capture.drc_strength = val
            self.logger.log(logging.DEBUG, "Status:DRC Strength:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set drc strength, camera not open!")

    @contrast.setter
    def contrast(self, val):
        if val is None: return
        val = int(val)
        if self.capture_open:
            with self.capture_lock: self.capture.contrast = val
            self.logger.log(logging.DEBUG, "Status:Contrast:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set contrast, camera not open!")
    #
    # Other Effects ####################################################
    #
    # flash_mode
    # clock mode "reset", is relative to start of recording, "raw" is relative to start of camera
    # color_effects, "None" or (u,v) where u and v are 0..255 e.g. (128,128) gives black and white image
    # flash_mode 'off', 'auto', 'on', 'redeye', 'fillin', 'torch' defaults is off
    # image_denoise, True or False, activates the denosing of the image
    # video_denoise, True or False, activates the denosing of the video recording
    # image_effect, can be negative, solarize, sketch, denoise, emboss, oilpaint, hatch, gpen, pastel, watercolor, film, blur, saturation, colorswap, washedout, colorpoint, posterise, colorbalance, cartoon, deinterlace1, deinterlace2, default is 'none'
    # image_effect_params, setting the parameters for the image effects see https://picamera.readthedocs.io/en/release-1.13/api_camera.html
    # sharpness -100..100 default 0
    # video_stabilization default is False

    # Read
    @property
    def flash_mode(self):            
        if self.capture_open: 
            return self.capture.flash_mode             
        else: return float('NaN')

    @property
    def clock_mode(self):            
        if self.capture_open: 
            return self.capture.clock_mode             
        else: return float('NaN')

    @property
    def sharpness(self):             
        if self.capture_open: 
            return self.capture.sharpness              
        else: return float('NaN')

    @property
    def color_effects(self):         
        if self.capture_open: 
            return self.capture.color_effects           
        else: return float('NaN')

    @property
    def image_effect(self):          
        if self.capture_open: 
            return self.capture.image_effect           
        else: return float('NaN')

    @property
    def image_denoise(self):         
        if self.capture_open: 
            return self.capture.image_denoise          
        else: return float('NaN')

    @property
    def video_denoise(self):         
        if self.capture_open: 
            return self.capture.video_denoise          
        else: return float('NaN')

    @property
    def video_stabilization(self):   
        if self.capture_open: 
            return self.capture.video_stabilization    
        else: return float('NaN')

    # Write

    @flash_mode.setter
    def flash_mode(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.flash_mode = val
            self.logger.log(logging.DEBUG, "Status:Flash Mode:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set flash mode, camera not open!")

    @clock_mode.setter
    def clock_mode(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.clock_mode = val
            self.logger.log(logging.DEBUG, "Status:Clock Mode:{}".format(val))
        else: 
            self.logger.log(logging.CRITICAL, "Status:Can not set capture clock, camera not open!")

    @sharpness.setter
    def sharpness(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.sharpness = val
            self.logger.log(logging.DEBUG, "Status:Sharpness:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set sharpness, camera not open!")

    @color_effects.setter
    def color_effects(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.color_effects = val
            self.logger.log(logging.DEBUG, "Status:Color Effects:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set color effects, camera not open!")

    @image_effect.setter
    def image_effect(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.image_effect = val
            self.logger.log(logging.DEBUG, "Status:Image Effect:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set image effect, camera not open!")

    @image_denoise.setter
    def image_denoise(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.image_denoise = val
            self.logger.log(logging.DEBUG, "Status:Image Denoise:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set image denoise, camera not open!")

    @video_denoise.setter
    def video_denoise(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.video_denoise = val
            self.logger.log(logging.DEBUG, "Status:Video Denoise:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set video denoise, camera not open!")

    @video_stabilization.setter
    def video_stabilization(self, val):
        if val is None:  return
        if self.capture_open:
            with self.capture_lock: self.capture.video_stabilization = val
            self.logger.log(logging.DEBUG, "Status:Video Stabilization:{}".format(val))
        else:
            self.logger.log(logging.CRITICAL, "Status:Can not set video stabilization, camera not open!")

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    print("Starting Capture")
    camera = piCapture(camera_num=0, res=(320,240))
    camera.start()

    window_handle = cv2.namedWindow("Pi Camera", cv2.WINDOW_AUTOSIZE)

    print("Getting Frames")
    while cv2.getWindowProperty("Pi Camera", 0) >= 0:
        if camera.new_frame:
            cv2.imshow('Pi Camera', camera.frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    camera.stop()
    cv2.destroyAllWindows()
