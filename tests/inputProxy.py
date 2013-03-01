from __future__ import print_function
import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.blend import *
from panvid.input import *
from panvid.inputProxy import *
import cv2


class TestProxy(unittest.TestCase):
    def testBorder(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(30, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        path = path[0:len(path)/2]
        spshpath =  PathFilters(path).applySpeedChange(speed=40).applyShake(shake=15).fixPath(frame_size,(img.shape[0],img.shape[1])).getPath()
        filt = VideoEffects()
        #filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30, filt=filt)
        inp = VideoInput("/tmp/test_speed_shake.avi")
        register = RegisterImagesDetect(inp.getClone())
        pred_path = register.getDiff("LK")
        proxy = StreamProxyBorder(inp.getClone(), borderWidth=13)
        blend = BlendOverlay2D(proxy.getClone())
        blend.blendNextN(pred_path,True,False)
        cv2.imwrite("/tmp/test.jpg", blend.getPano())


