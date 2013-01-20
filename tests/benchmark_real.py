from panvid.generateVideo import *
from panvid.generatePath import *
from panvid.predict import *
from panvid.input import *
import cv2
import cProfile
import pstats

class Benchmark(object):
    def __init__(self, vidpath="tests/samples/DSC_0004.MOV"):
        self.vidpath = vidpath

    def bench_method(self, method="SIFT"):
        register = RegisterImagesDetect(VideoInput(self.vidpath))
        pred_path = register.getDiff2D(method)
        diff = (0,0)
        rdiff = (0,0)
        drift = (0,0)
        rdrift = (0,0)
        badpoints = 0
        """for (p,r) in zip(pred_path, self.npath):
            if p is not None:
                q, p = p
                diff = (diff[0] + abs(p[0] - r[0]),
                        diff[1]+  abs(p[1] - r[1]))
                rdiff = (rdiff[0] + abs(round(p[0]) - r[0]),
                         rdiff[1] + abs(round(p[1]) - r[1]))
                drift = (drift[0] + (p[0] - r[0]),
                         drift[1] + (p[1] - r[1]))
                rdrift =(rdrift[0] + (round(p[0]) - r[0]),
                         rdrift[1] + (round(p[1]) - r[1]))
                if abs(round(p[0]) - r[0]) + abs(round(p[1]) - r[1]) != 0:
                    badpoints += 1

            else:
                badpoints += 1"""
        return (diff, rdiff, drift, rdrift, badpoints)

b = Benchmark()
retval = []
for vidpath in ["tests/samples/DSC_0004.MOV"]:
    b = Benchmark(vidpath)
    for method in ["LK-SIFT", "LK-SURF", "LK", "SIFT", "SURF"]:
        print "Calculating for method " + method + " " +vidpath
        cProfile.runctx("retval = b.bench_method('%s')" % method, locals(), globals(), "/tmp/bench")
        print "returned " + str(retval)
        p = pstats.Stats("/tmp/bench")
        p.strip_dirs().sort_stats("time").print_stats(3)
        print ""
