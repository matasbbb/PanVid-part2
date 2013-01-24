
import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.blend import *
from panvid.input import *
from panvid.warp import *
import math
import cv2


class TestHomography(unittest.TestCase):
    def test_homoFocalTest_realVid(self):
        vidinput = VideoInputSkip("tests/samples/MVI_0017.AVI", 0, 10)
        register = RegisterImagesDetect(vidinput)
        homos = register.getDiff("LK-SURF")
        for h in homos:
            if h[1] is not None:
                (f1, f2) = focalFromHomography(h[1])
                if f1 is not None and f2 is not None:
                    print (h[0],(f1, f2), math.sqrt(f1 * f2))
                else:
                    if f1 is None:
                        f1 = 1
                    if f2 is None:
                        f2 = 1
                    print (h[0],(f1, f2), math.sqrt(f1 * f2), "Fake")
            else:
                print "None"
