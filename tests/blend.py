from __future__ import print_function
import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.blend import *
from panvid.input import *
import cv2


class TestBlend(unittest.TestCase):
    synteticInput = None
    synteticDataPoints = None
    realInput = None
    realDataPoints = None
    @classmethod
    def setUpClass(self):
        progressCB = lambda *args: print (args)
        if self.synteticDataPoints == None:
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
            register = RegisterImagesDetect(VideoInput(vidpath))
            self.synteticInput = VideoInput(vidpath)
            self.synteticDataPoints = register.getDiff("LK-SURF", progressCB=progressCB)
        if self.realDataPoints == None:
            self.realInput = VideoInputAdvanced("tests/samples/MVI_0017.AVI",
                    bound=10)
            register = RegisterImagesDetect(self.realInput.getClone())
            self.realDataPoints = register.getDiff("LK-SURF",0.8, progressCB=progressCB)

    def non_test_overlay2d(self):
        blend = BlendOverlay2D(self.synteticInput.getClone())
        blend.blendNextN(self.synteticDataPoints, True, False)
        blend.prevPano()

    def test_overlay2d_realVid(self):
        blend = BlendOverlay2D(self.realInput.getClone())
        blend.blendNextN(self.realDataPoints, prev=True, wait=False)
        blend.prevPano(wait=False, window="2D Real")
        cv2.imwrite("/tmp/test.jpg", blend.getPano())


    def non_test_overlayHomo(self):
        blend = BlendOverlayHomo(self.synteticInput.getClone())
        blend.blendNextN(self.synteticDataPoints, True)

    def test_overlayHomo_realVid(self):
        blend = BlendOverlayHomoWeight(self.realInput.getClone())
        blend.blendNextN(self.realDataPoints, True, False)
        blend.prevPano(wait=False, window="Homo Weight")
        cv2.imwrite("/tmp/testSecond.jpg", blend.getPano())

