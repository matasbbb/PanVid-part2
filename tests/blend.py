import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.blend import *
from panvid.input import *
import cv2


class TestBlend(unittest.TestCase):
    def not_test_overlay2d(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(10, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().fixPath(frame_size,(img.shape[0], img.shape[1])).getPath()
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30, filt=filt)
        register = RegisterImagesDetect(VideoInput("/tmp/test_speed_shake.avi"))
        dataPoints = register.getDiff("LK-SURF")
        blend = BlendOverlay2D(VideoInput("/tmp/test_speed_shake.avi"))
        blend.blendNextN(dataPoints, True, False)
        blend.prevPano()

    def not_test_overlay2d_realVid(self):
        stream = VideoInputSkip("tests/samples/MVI_0017.AVI", 0, 3)
        register = RegisterImagesDetect(stream.getClone())
        dataPoints = register.getDiff("LK-SURF",0.7)
        blend = BlendOverlay2D(stream.getClone())
        blend.blendNextN(dataPoints, True, False)
        blend.prevPano()
        cv2.imwrite("/tmp/test.jpg", blend.getPano())


    def nottest_overlayHomo(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(10, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().fixPath(frame_size,(img.shape[0],img.shape[1])).getPath()
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30, filt=filt)
        register = RegisterImagesDetect(VideoInput("/tmp/test_speed_shake.avi"))
        dataPoints = register.getDiff("LK-SURF")
        blend = BlendOverlayHomo(VideoInput("/tmp/test_speed_shake.avi"))
        blend.blendNextN(dataPoints, True)

    def test_overlayHomo_realVid(self):
        stream = VideoInputSkip("tests/samples/MVI_0017.AVI", 300, 0)
        register = RegisterImagesDetect(stream)
        homos = register.getDiff("LK-SURF", 0.8)
        for h in homos:
            print str(h)
        blend = BlendOverlayHomoBorders(stream.getClone())
        blend.blendNextN(homos, True, False)
        blend.prevPano()
        cv2.imwrite("/tmp/testSecond.jpg", blend.getPano())

