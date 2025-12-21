###############################################################################
# Raspberry Pi CSI video capture
# Uses the picamera2 library to interface with the Raspberry Pi Camera Module
#
# Allows configuration of codec
# Urs Utzinger 2019, 2020, 2025
###############################################################################

###############################################################################
# Imports
###############################################################################

# Multi Threading
from threading import Thread, Lock
from queue import Queue

#
import logging, time, sys

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
    ############################################################################
    def __init__(self, configs, 
        camera_num: int = 0, 
        res: tuple = None, 
        exposure: float = None,
        queue_size: int = 32):

        # Proper Thread initialization (so .start()/.join() work as expected)
        super().__init__(daemon=True)

        # Keep a copy of configs for consistency across capture modules
        self._configs = configs or {}

        # populate desired settings from configuration file or function call
        ############################################################################
        self.camera_num      = camera_num
        if exposure is not None:
            self._exposure   = exposure  
        else: 
            self._exposure   = self._configs.get('exposure', -1)
        if res is not None:
            self._camera_res = res
        else: 
            self._camera_res = self._configs.get('camera_res', (640, 480))
        self._capture_width  = self._camera_res[0] 
        self._capture_height = self._camera_res[1]
        self._output_res     = self._configs.get('output_res', (-1, -1))
        self._output_width   = self._output_res[0]
        self._output_height  = self._output_res[1]
        self._framerate      = self._configs.get('fps', 30)
        self._flip_method    = self._configs.get('flip', 0)

        # Threading Queue, Locks, Events
        self.capture         = Queue(maxsize=queue_size)
        self.log             = Queue(maxsize=32)
        self.stopped         = True

        # open up the camera
        self.open_cam()

        # Init vars
        self.frame_time   = 0.0
        self.measured_fps = 0.0

    # Thread routines #################################################
    # Start Stop and Update Thread
    ###################################################################

    def stop(self):
        """stop the thread"""
        self.stopped = True
        # clrean up

    def close_cam(self):
        """Close stream/buffers/camera (idempotent)."""
        try:
            stream = getattr(self, 'stream', None)
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self.stream = None

            raw = getattr(self, 'rawCapture', None)
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    pass
            self.rawCapture = None

            cam = getattr(self, 'cam', None)
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
            self.cam = None
            self.cam_open = False
        except Exception:
            pass

    def start(self):
        """start the capture thread"""
        self.stopped = False
        super().start()

    # Thread entrypoint
    def run(self):
        """thread entrypoint"""
        self.update()

    # After Stating of the Thread, this runs continously
    def update(self):
        """ run the thread """
        try:
            self.stream = self.cam.capture_continuous(
                self.rawCapture,
                format="bgr", 
                use_video_port=True)
        except:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Failed to create camera stream!"))

        last_time = time.time()
        num_frames = 0

        for f in self.stream:
            current_time = time.time()

            if self.stopped:
                break

            img = f.array
            self.rawCapture.truncate(0)
            num_frames += 1
            self.frame_time = int(current_time*1000)

            if (img is not None) and (not self.capture.full()):

                img_proc = img

                # Resize only if an explicit output size was provided
                if (self._output_width > 0) and (self._output_height > 0):
                    img_proc = cv2.resize(img_proc, self._output_res)

                # Apply flip/rotation if requested (same enum as cv2Capture)
                if self._flip_method != 0:
                    if self._flip_method == 1: # ccw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self._flip_method == 2: # rot 180
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_180)
                    elif self._flip_method == 3: # cw 90
                        img_proc = cv2.rotate(img_proc, cv2.ROTATE_90_CLOCKWISE)
                    elif self._flip_method == 4: # horizontal (left-right)
                        img_proc = cv2.flip(img_proc, 1)
                    elif self._flip_method == 5: # upright diagonal. ccw & lr
                        img_proc = cv2.flip(cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
                    elif self._flip_method == 6: # vertical (up-down)
                        img_proc = cv2.flip(img_proc, 0)
                    elif self._flip_method == 7: # upperleft diagonal
                        img_proc = cv2.transpose(img_proc)

                self.capture.put_nowait((self.frame_time, img_proc))
            else:
                if not self.log.full(): self.log.put_nowait((logging.WARNING, "PiCap:Capture Queue is full!"))

            # FPS calculation
            if (current_time - last_time) >= 5.0: # update frame rate every 5 secs
                self.measured_fps = num_frames/5.0
                if not self.log.full(): self.log.put_nowait((logging.INFO, "PICAM:FPS:{}".format(self.measured_fps)))
                num_frames = 0
                last_time = current_time

        self.close_cam()

    # Setup the Camera
    ############################################################################
    def open_cam(self):
        """
        Open up the camera so we can begin capturing frames
        """
        
        # Open PiCamera
        ###############
        self.cam      = PiCamera(self.camera_num)
        self.cam_open = not self.cam.closed

        if self.cam_open:
            # Apply settings to camera
            self.resolution    = self._camera_res
            self.rawCapture    = PiRGBArray(self.capture)
            self.fps           = self._framerate
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
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Failed to open camera!"))

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
        if self.cam_open: 
            return self.cam.resolution                            
        else: return float("NaN")

    @property
    def width(self):                 
        if self.cam_open: 
            return self.cam.resolution[0]
        else: return float("NaN")

    @property
    def height(self):                
        if self.cam_open: 
            return self.cam.resolution[1]                         
        else: return float("NaN")

    @property
    def fps(self):             
        if self.cam_open: 
            return self.cam.framerate+self.cam.framerate_delta 
        else: return float("NaN")

    @property
    def exposure(self):              
        if self.cam_open: 
            return self.cam.exposure_speed                        
        else: return float("NaN")

    # Write
    ###################################################################

    @resolution.setter
    def resolution(self, val):
        if val is None: return
        if self.cam_open:
            if len(val) > 1:
                self.cam.resolution = val
                if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Width:{},Height:{}".format(val[0],val[1])))
                self._resolution = val
            else:
                self.cam.resolution = (val, val)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Width:{},Height:{}".format(val,val)))
                self._resolution = (val, val)                    
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set resolution, camera not open!"))

    @width.setter
    def width(self, val):
        if val is None: return
        val = int(val)
        if self.cam_open:
            self.cam.resolution = (val, self.cam.resolution[1])
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Width:{}".format(val)))

        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set resolution, camera not open!"))

    @height.setter
    def height(self, val):
        if val is None: return
        val = int(val)
        if self.cam_open:
            self.cam.resolution = (self.cam.resolution[0], val)
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Height:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set resolution, camera not open!"))

    @fps.setter
    def fps(self, val):
        if val is None: return
        val = float(val)
        if self.cam_open:
            self.cam.framerate = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:FPS:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set franerate, camera not open!"))

    @exposure.setter
    def exposure(self, val):
        if val is None: return
        if self.cam_open:
            self.cam.shutter_speed  = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Exposure:{}".format(val)))
            self._exposure = self.exposure
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set exposure, camera not open!"))

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
        if self.cam_open: 
            return self.cam.awb_mode               
        else: return float('NaN')

    @property
    def awb_gains(self):             
        if self.cam_open: 
            return self.cam.awb_gains              
        else: return float('NaN')

    @property
    def analog_gain(self):           
        if self.cam_open: 
            return self.cam.analog_gain           
        else: return float("NaN")

    @property
    def digital_gain(self):          
        if self.cam_open: 
            return self.cam.digital_gain           
        else: return float("NaN")

    # Write

    @awb_mode.setter
    def awb_mode(self, val):
        if val is None: return
        if self.cam_open:
            self.cam.awb_mode  = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:AWB Mode:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set autowb, camera not open!"))
 
    @awb_gains.setter
    def awb_gains(self, val):
        if val is None: return
        if self.cam_open:
            if len(val) > 1:
                self.cam.awb_gains  = val
                if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:AWB Gains:red:{},blue:{}".format(val[0], val[1])))
            else:
                self.cam.awb_gains = (val, val)
                if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:AWB Gain:{},{}".format(val,val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set autowb gains, camera not open!"))

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
        if self.cam_open: 
            return self.cam.brightness             
        else: return float('NaN')

    @property
    def iso(self):                   
        if self.cam_open: 
            return self.cam.iso                    
        else: return float("NaN")

    @property
    def exposure_mode(self):         
        if self.cam_open: 
            return self.cam.exposure_mode          
        else: return float("NaN")

    @property
    def exposure_compensation(self): 
        if self.cam_open: 
            return self.cam.exposure_compensation  
        else: return float("NaN")

    @property
    def drc_strength(self):          
        if self.cam_open: 
            return self.cam.drc_strength           
        else: return float('NaN')

    @property
    def contrast(self):              
        if self.cam_open: 
            return self.cam.contrast               
        else: return float('NaN')

    # Write

    @brightness.setter
    def brightness(self, val):
        if val is None:  return
        val = int(val)
        if self.cam_open:
            self.cam.brightness = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Brightness:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set brightnes, camera not open!"))

    @iso.setter
    def iso(self, val):
        if val is None: return
        val = int(val)
        if self.cam_open:
            self.cam.iso = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:ISO:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set ISO, camera not open!"))

    @exposure_mode.setter
    def exposure_mode(self, val):
        if val is None: return
        if self.cam_open:
            self.cam.exposure_mode = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Exposure Mode:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set exposure mode, camera not open!"))

    @exposure_compensation.setter
    def exposure_compensation(self, val):
        if val is None: return
        val = int(val)
        if self.cam_open:
            self.cam.exposure_compensation = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Exposure Compensation:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set exposure compensation, camera not open!"))

    @drc_strength.setter
    def drc_strength(self, val):
        if val is None: return
        if self.cam_open:
            self.cam.drc_strength = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:DRC Strength:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set drc strength, camera not open!"))

    @contrast.setter
    def contrast(self, val):
        if val is None: return
        val = int(val)
        if self.cam_open:
            self.cam.contrast = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Contrast:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set contrast, camera not open!"))

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
        if self.cam_open: 
            return self.cam.flash_mode             
        else: return float('NaN')

    @property
    def clock_mode(self):            
        if self.cam_open: 
            return self.cam.clock_mode             
        else: return float('NaN')

    @property
    def sharpness(self):             
        if self.cam_open: 
            return self.cam.sharpness              
        else: return float('NaN')

    @property
    def color_effects(self):         
        if self.cam_open: 
            return self.cam.color_effects           
        else: return float('NaN')

    @property
    def image_effect(self):          
        if self.cam_open: 
            return self.cam.image_effect           
        else: return float('NaN')

    @property
    def image_denoise(self):         
        if self.cam_open: 
            return self.cam.image_denoise          
        else: return float('NaN')

    @property
    def video_denoise(self):         
        if self.cam_open: 
            return self.cam.video_denoise          
        else: return float('NaN')

    @property
    def video_stabilization(self):   
        if self.cam_open: 
            return self.cam.video_stabilization    
        else: return float('NaN')

    # Write

    @flash_mode.setter
    def flash_mode(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.flash_mode = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Flash Mode:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set flash mode, camera not open!"))

    @clock_mode.setter
    def clock_mode(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.clock_mode = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Clock Mode:{}".format(val)))
        else: 
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set capture clock, camera not open!"))

    @sharpness.setter
    def sharpness(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.sharpness = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Sharpness:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set sharpness, camera not open!"))

    @color_effects.setter
    def color_effects(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.color_effects = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Color Effects:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set color effects, camera not open!"))

    @image_effect.setter
    def image_effect(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.image_effect = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Image Effect:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set image effect, camera not open!"))

    @image_denoise.setter
    def image_denoise(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.image_denoise = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Image Denoise:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set image denoise, camera not open!"))

    @video_denoise.setter
    def video_denoise(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.video_denoise = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Video Denoise:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set video denoise, camera not open!"))

    @video_stabilization.setter
    def video_stabilization(self, val):
        if val is None:  return
        if self.cam_open:
            self.cam.video_stabilization = val
            if not self.log.full(): self.log.put_nowait((logging.INFO, "PiCap:Video Stabilization:{}".format(val)))
        else:
            if not self.log.full(): self.log.put_nowait((logging.CRITICAL, "PiCap:Can not set video stabilization, camera not open!"))

###############################################################################
# Testing
###############################################################################

if __name__ == '__main__':

    configs = {
        'camera_res'      : (1280, 720),    # any amera: Camera width & height
        'exposure'        : 10000,          # any camera: -1,0 = auto, 1...max=frame interval in microseconds
        'autoexposure'    : 0,              # cv2 camera only, depends on camera: 0.25 or 0.75(auto), -1,0,1
        'fps'             : 60,             # any camera: 1/10, 15, 30, 40, 90, 120 overlocked
        'output_res'      : (-1, -1),       # Output resolution 
        'flip'            : 0,              # 0=norotation 
                                            # 1=ccw90deg 
                                            # 2=rotation180 
                                            # 3=cw90 
                                            # 4=horizontal 
                                            # 5=upright diagonal flip 
                                            # 6=vertical 
                                            # 7=uperleft diagonal flip
        'displayfps'       : 30             # frame rate for display server
    }

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("Capture")

    logger.log(logging.DEBUG, "Starting Capture")

    camera = piCapture(configs, camera_num=0)
    camera.start()

    logger.log(logging.DEBUG, "Starting Capture")

    window_handle = cv2.namedWindow("Pi Camera", cv2.WINDOW_AUTOSIZE)
    while cv2.getWindowProperty("Pi Camera", 0) >= 0:
        try:
            (frame_time, frame) = camera.capture.get()
            cv2.imshow('Pi Camera', frame)
        except: pass

        if cv2.waitKey(1) & 0xFF == ord('q'):  break

        try: 
            (level, msg)=camera.log.get_nowait()
            logger.log(level, "PiCap:{}".format(msg))
        except: pass

    camera.stop()
    cv2.destroyAllWindows()