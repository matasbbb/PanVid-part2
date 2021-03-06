from panvid.generateVideo import *
from panvid.generatePath import *
from panvid.predict import *
from panvid.input import *
import cv2
import cProfile
import pstats

class Benchmark(object):
    def __init__(self, imgpath="tests/samples/IMG_8686.JPG", vidpath="/tmp/bench_%s.avi", frame_size=(1000,1000)):
        img = cv2.imread(imgpath)
        gen = PathGenerator(120, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().fixPath(frame_size, (img.shape[0], img.shape[1])).getPath()
        #spshpath = path
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidpath = vidpath % str(frame_size)
        self.vidpath = vidpath
        vidgen.save(vidpath, frame_size, fps=30, filt=filt)
        self.npath = []
        lt = spshpath[0]
        for t in spshpath[1:]:
            self.npath.append((round(t[0])-round(lt[0]),round(t[1])-round(lt[1])))
            lt = t

    def bench_method(self, method="SIFT"):
        register = RegisterImagesDetect(VideoInput(self.vidpath))
        pred_path = register.getDiff(method)
        diff = (0,0)
        rdiff = (0,0)
        drift = (0,0)
        rdrift = (0,0)
        badpoints = 0
        for (p,r) in zip(pred_path, self.npath):
            if p is not None:
                p = p.get_naive2D
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
                badpoints += 1
        return (diff, rdiff, drift, rdrift, badpoints)

b = Benchmark()
retval = []
for frame_size in [(500,2000),(1000,1000)]:
    b = Benchmark(frame_size=frame_size)
    for method in ["LK", "SIFT", "SURF"]:
        print "Calculating for method " + method + str(frame_size)
        cProfile.runctx("retval = b.bench_method('%s')" % method, locals(), globals(), "/tmp/bench")
        print "returned " + str(retval)
        p = pstats.Stats("/tmp/bench")
        p.strip_dirs().sort_stats("time").print_stats(3)
        print ""
