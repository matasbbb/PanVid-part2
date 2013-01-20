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

        return "Worked"

b = Benchmark()
retval = []
for vidpath in ["tests/samples/DSC_0004.MOV"]:
    b = Benchmark(vidpath)
    for method in ["LK", "LK-SIFT", "LK-SURF", "SIFT", "SURF"]:
        print "Calculating for method " + method + " " +vidpath
        cProfile.runctx("retval = b.bench_method('%s')" % method, locals(), globals(), "/tmp/bench")
        print "returned " + str(retval)
        p = pstats.Stats("/tmp/bench")
        p.strip_dirs().sort_stats("time").print_stats(3)
        print ""
