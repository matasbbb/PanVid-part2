import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.input import *
import cv2


class TestRegisterImages(unittest.TestCase):
    def test_getPath(self):
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
        npath = []
        lt = spshpath[0]
        for t in spshpath[1:]:
            npath.append((round(t[0])-round(lt[0]),round(t[1])-round(lt[1])))
            lt = t

        print "SURF"
        pred_path = register.getDiff2D("SURF")
        diff = (0,0)
        drift = (0,0)
        for (p,r) in zip(pred_path, npath):
            diff = (diff[0] + abs(p[0] - r[0]), diff[1]+  abs(p[1] - r[1]))
            drift = (drift[0] + (p[0] - r[0]), drift[1] + (p[1] - r[1]))
            print str(p) + str(r)
        print diff
        print drift
        print("SIFT")
        register = RegisterImagesStandart2D(VideoInput("/tmp/test_speed_shake.avi"))
        pred_path = register.getDiff2D("SIFT")
        diff = (0,0)
        drift = (0,0)
        for (p,r) in zip(pred_path, npath):
            diff = (diff[0] + abs(p[0] - r[0]), diff[1]+  abs(p[1] - r[1]))
            drift = (drift[0] + (p[0] - r[0]), drift[1] + (p[1] - r[1]))
            print str(p) + str(r)
        print diff
        print drift

