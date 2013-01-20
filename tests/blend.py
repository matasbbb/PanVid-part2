import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.blend import *
from panvid.input import *
import cv2


class TestBlend(unittest.TestCase):
    def test_overlay2d(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(10, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().getPath()
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30, filt=filt)
        register = RegisterImagesStandart2D(VideoInput("/tmp/test_speed_shake.avi"))
        shifts = register.getDiff2D("SURF")
        blend = BlendOverlay2D(VideoInput("/tmp/test_speed_shake.avi"))
        blend.blendNextN(shifts, True)

    def test_overlay2d_realVid(self):
        stream = VideoInputSkip("tests/samples/DSC_0004.MOV", 40, 3)
        register = RegisterImagesStandart2D(stream.getClone())
        shifts = register.getDiff2D("SURF")
        blend = BlendOverlay2D(stream.getClone())
        blend.blendNextN(shifts, True)

    def test_overlayHomo(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(10, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().getPath()
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30, filt=filt)
        register = RegisterImagesStandart2D(VideoInput("/tmp/test_speed_shake.avi"))
        homos = register.getDiff("SURF")
        blend = BlendOverlayHomo(VideoInput("/tmp/test_speed_shake.avi"))
        blend.blendNextN(homos, True)

    def overlayHomo_realVid(self):
        register = RegisterImagesStandart2D(VideoInputSkip("tests/samples/DSC_0004.MOV", 40, 1))
        homos = register.getDiff("SURF")
        blend = BlendOverlayHomo(VideoInputSkip("tests/samples/DSC_0004.MOV", 40, 1))
        blend.blendNextN(homos, True)

