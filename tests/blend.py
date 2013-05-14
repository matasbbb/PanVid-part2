from __future__ import print_function
import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.blend import *
from panvid.input import *
from panvid.inputProxy import *
import numpy as np
import time
import cv2


class TestBlend(unittest.TestCase):
    synteticInput = None
    synteticDataPoints = None
    realInput = None
    realDataPoints = None
    @classmethod
    def setUpClass(self):
        progressCB = lambda *args: print (args)
        if self.synteticDataPoints == None and False:
            frame_size = (1000,1000)
            img = cv2.imread("tests/samples/IMG_8686.JPG")
            gen = PathGenerator(30, img, None, frame_size)
            path = gen.getSweapPath(2000, False)
            spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().fixPath(frame_size,(img.shape[0], img.shape[1])).getPath()
            filt = VideoEffects()
            filt.noise()
            vidgen = VideoSimpleGenerator(spshpath, img)
            vidpath = "/tmp/test_speed_shake.avi"
            vidgen.save(vidpath, frame_size, fps=30, filt=filt)
            register = RegisterImagesContByString(VideoInput(vidpath), "LK-SURF")
            self.synteticInput = VideoInput(vidpath)
            self.synteticDataPoints = register.getDiff(progressCB=progressCB)
        if self.realDataPoints == None:
            path = "tests/samples/MVI_0017.AVI"
            #path = "tests/samples/DSC_0011.MOV"
            self.realInput = VideoInputAdvanced(path, bound=100)
            register = RegisterImagesContByString(self.realInput.getClone(), "LK-SURF")
            #self.realInput = StreamProxyBorder(self.realInput, borderWidth=1)
            #self.realDataPoints = register.getDiff(progressCB=progressCB)

    def non_test_overlay2d(self):
        blend = BlendOverlay2D(self.synteticInput.getClone())
        blend.blendNextN(self.synteticDataPoints, True, False)
        blend.prevPano()

    def non_test_overlay2d_realVid(self):
        blend = BlendOverlay2D(self.realInput.getClone())
        blend.blendNextN(self.realDataPoints, prev=True, wait=False)
        blend.prevPano(wait=True, window="2D Real")
        cv2.imwrite("/tmp/test.jpg", blend.getPano())


    def non_test_overlayHomo(self):
        blend = BlendOverlayHomo(self.synteticInput.getClone())
        blend.blendNextN(self.synteticDataPoints, True)
        blend.prevPano(wait=True, window="Homo Overlay")

    def non_test_overlayHomo_realVid(self):
        blend = BlendOverlayHomo(self.realInput.getClone())
        blend.blendNextN(self.realDataPoints, prev=True, wait=False)
        blend.prevPano(wait=False, window="Homo Overlay")
        cv2.imwrite("/tmp/testSecond.jpg", blend.getPano())

    def non_test_contin_realVid(self):
        progressCB = lambda *args: print (args)
        reg = RegisterImagesContByString(self.realInput.getClone(),"LK-SURF")
        blend = BlendOverlayHomo(None, reg, progressCB)
        blend.blendNextN(None, False, False)
        cv2.imwrite("/tmp/testSecondLK.jpg", blend.getPano())
    
    def non_test_masks(self):
        for name in BlendRegister.keys():
            blendm, _ = BlendRegister[name]
            path = "tests/samples/moresample/0.3MpxParkCenterSlow360,21sec,SunDist.AVI"
            b = blendm(VideoInput(path))
            mask = b._mask
            minimal = mask.min()*1.0
            maximal = mask.max()*1.0
            mask = (((mask.astype(np.float) - minimal)/maximal)*256).astype(np.uint8)
            cv2.imwrite("/media/sf_G_DRIVE/new/images/mask" + name +".jpg", mask)
            print(name)


    def test_all_samples(self):
        path = "tests/samples/moresample/"
        samples = [("1flare", path + "0.3MpxParkCenterSlow360,21sec,SunDist.AVI", 210, 265),["2verticalgood", path + "2MpxParkSide320,73sec,Sideways.MOV", 500,600],["3normal",path+"2MpxParkSide320,20sec,LittleJumpy.MOV", 500, 600], ["4normal",path+"2MpxParkSide320,20sec,LittleJumpy.MOV", 107, 109]]
        for sample_name, path, start, end in samples[3:4]:
            stream = VideoInputAdvanced(path, end, 0, start)
            register = RegisterImagesContJumped(stream.getClone(), 3, method="LK", quality=0.90, jumpmethod="SURF")
            pred_path = register.getDiff()
            for warp in [False, True]:
                for name in BlendRegister.keys():
                    blendm, _ = BlendRegister[name]
                    pstream = stream.getClone()
                    #pstream = StreamProxyBorder(pstream, random=True)
                    pstream = StreamProxyColor(pstream) 
                    blend = blendm(pstream, warp=warp)
                    sec = 0 - time.time()
                    blend.blendNextN(pred_path, False, False)
                    sec += time.time()
                    cv2.imwrite("/media/sf_G_DRIVE/new/images/time" +
                                sample_name + " " + name + " " + 
                                str(warp) + str(sec)+".jpg",
                                blend.getPano())

    def not_test_qualitive_samples(self):
        path = "tests/samples/moresample/"
        samples = [("1flare", path + "0.3MpxParkCenterSlow360,21sec,SunDist.AVI", 210, 265),
                ["2verticalgood", path + "2MpxParkSide320,73sec,Sideways.MOV", 500,600],
                ["3normal",path+"2MpxParkSide320,20sec,LittleJumpy.MOV", 500, 600], 
                ["4normal",path+"2MpxParkSide320,20sec,LittleJumpy.MOV", 100, 120],
                ["5shake",path+"2MpxParkSide90,24sec,VeryShaky.MOV",0,300]]
        for sample_name, path, start, end in samples[4:5]:
            stream = VideoInputAdvanced(path, end, 0, start)
            for warp in [False, True]:
                for q in [0., 0.1,0.4,0.7,0.85,0.9,0.95, 0.98]:
                    blendm, _ = BlendRegister["Overlay nearest3"]
                    register = RegisterImagesContJumped(stream.getClone(), 3, method="LK", quality=q, jumpmethod="SURF")
                    pred_path = register.getDiff()
                    pstream = stream.getClone()
                    count = 0
                    for p in pred_path:
                        if p.get_homo() is None:
                            count += 1
                    print (count)

                    worse = 0
                    for p in pred_path:
                        if p.get_quality() < q:
                            worse += 1
                    print (worse)

                    #pstream = StreamProxyBorder(pstream, random=True) 
                    blend = blendm(pstream, warp=warp)
                    sec = 0 - time.time()
                    blend.blendNextN(pred_path, False, False)
                    sec += time.time()
                    cv2.imwrite("/media/sf_G_DRIVE/new/images/quality" +
                                sample_name + " " + str(q) + " " + 
                                str(count) + " " + str(worse) + " " +
                                str(warp) + str(sec)+".jpg",
                                blend.getPano())

           
    def non_test_contin_webcam(self):
        progressCB = lambda *args: print (args)
        reg = RegisterImagesContByString(VideoInput(0),"LK-SURF")
        blend = BlendOverlayHomo(None, reg, progressCB)
        blend.blendNextN(None, True, False)

    def non_test_overlayHomo_realVid(self):
        blend = BlendOverlayHomoWeight(self.realInput.getClone())
        blend.blendNextN(self.realDataPoints, True, False)
        blend.prevPano(wait=False, window="Homo Weight")
        cv2.imwrite("/tmp/testSecond.jpg", blend.getPano())

