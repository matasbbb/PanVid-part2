import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.input import *
import cv2


class TestRegisterImages(unittest.TestCase):
    def test_syntheticGetPath(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(10, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().fixPath(frame_size,(img.shape[0],img.shape[1])).getPath()
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30, filt=filt)
        npath = []
        lt = spshpath[0]
        for t in spshpath[1:]:
            npath.append((round(t[0])-round(lt[0]),round(t[1])-round(lt[1])))
            lt = t

        inp = VideoInput("/tmp/test_speed_shake.avi")
        for m in ["LK", "LK-SURF", "SURF"]:
            print m
            register = RegisterImagesContByString(inp.getClone(), m)
            pred_path = register.getDiff()
            diff = (0,0)
            drift = (0,0)
            sqdiff = 0
            self.assertEqual(len(pred_path), len(npath))
            for (pd,r) in zip(pred_path, npath):
                if pd is not None and r is not None:
                    p = pd.get_naive2D()
                    diff = (diff[0] + abs(p[0] - r[0]), diff[1]+  abs(p[1] - r[1]))
                    drift = (drift[0] + (p[0] - r[0]), drift[1] + (p[1] - r[1]))
                    sqdiff += (p[0] - r[0])**2 + (p[1] - r[1])**2
                print str(p) + str(r) + " " + str(pd)
            print diff
            print drift
            print ""

    def test_getCompWithSurfForLive(self):
        inp = VideoInputAdvanced("tests/samples/DSC_0004.MOV", 40, 0)
        register = RegisterImagesContByString(inp.getClone(), "SURF")
        npath = register.getDiff()

        for m in ["LK", "LK-SURF", "SURF"]:
            print "\n"+m
            register = RegisterImagesContByString(inp.getClone(), m)
            pred_path = register.getDiff()
            diff = 0
            self.assertEqual(len(pred_path), len(npath))
            for (p,r) in zip(pred_path, npath):
                df = p.get_distance(r)
                print (df, str(p))
                if df is not None:
                    diff += df
            print diff
            print ""

