import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
from panvid.predict import *
from panvid.input import *
import cv2


class TestRegisterImages(unittest.TestCase):
    def not_test_getPath(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(10, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().getPath()
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
            register = RegisterImagesDetect(inp.getClone())
            pred_path = register.getDiff2D(m)
            diff = (0,0)
            drift = (0,0)
            sqdiff = 0
            self.assertEqual(len(pred_path), len(npath))
            for (p,r) in zip(pred_path, npath):
                if p is not None and r is not None:
                    q, p = p
                    diff = (diff[0] + abs(p[0] - r[0]), diff[1]+  abs(p[1] - r[1]))
                    drift = (drift[0] + (p[0] - r[0]), drift[1] + (p[1] - r[1]))
                    sqdiff += (p[0] - r[0])**2 + (p[1] - r[1])**2
                print str(p) + str(r) + " " + str(q)
            print diff
            print drift
            print ""

    def test_getCompWithSurfForLive(self):
        inp = VideoInputSkip("tests/samples/DSC_0004.MOV", 40, 1)
        register = RegisterImagesDetect(inp.getClone())
        npath = register.getDiff("SURF")

        for m in ["LK", "LK-SIFT", "SIFT", "SURF"]:
            print "\n"+m
            register = RegisterImagesDetect(inp.getClone())
            pred_path = register.getDiff(m)
            diff = 0
            self.assertEqual(len(pred_path), len(npath))
            cor = np.array([[0,0],[0,1000],[1000,0],[1000,1000]],dtype='float32')
            cor = np.array([cor])
            for (p,r) in zip(pred_path, npath):
                if p is not None and r is not None:
                    q, p = p
                    _, r = r
                    df = 0
                    des0 = cv2.perspectiveTransform(cor,p)
                    des1 = cv2.perspectiveTransform(cor,r)

                    for p1,p2 in zip(des0[0],des1[0]):
                        df += abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
                    print df
                    diff += df
            print diff
            print ""

