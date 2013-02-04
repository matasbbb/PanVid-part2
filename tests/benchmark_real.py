from panvid.generateVideo import *
from panvid.generatePath import *
from panvid.predict import *
from panvid.input import *
from panvid.pyprof2calltree import *
import cv2
import cProfile
import pstats
intrest_code = \
    {'<cv2.findHomography>':("RANSAC", 0),
     '<cv2.goodFeaturesToTrack>':("LK Feature", 0),
     '<cv2.calcOpticalFlowPyrLK>':("LK Compute", 0),
     "<method 'compute' of 'cv2.DescriptorExtractor' objects>":("%s Compute", 1),
     "<method 'detect' of 'cv2.FeatureDetector' objects>":("%s Feature", 1)
    }

def create_cvs(data):
    rows = {"Data":[]}
    val = 0
    for method in data.keys():
        d = data[method]
        d["Data"] = method
        for instr in d.keys():
            if not rows.has_key(instr):
                rows[instr] = [0]*val
        for instr in rows.keys():
            if d.has_key(instr):
                rows[instr].append(d[instr])
            else:
                rows[instr].append(0)
        val += 1
    ret_string = ""
    for r in rows.keys():
        ret_string += r + ", " + ", ".join(map(str,rows[r])) + "\n"
    print ret_string


class Benchmark(object):
    def __init__(self, vidpath="tests/samples/DSC_0004.MOV"):
        self.vidpath = vidpath

    def bench_method(self, method="SIFT", skip=0):
        stream = VideoInputSkip(self.vidpath,skip=skip)
        register = RegisterImagesDetect(stream)
        pred_path = register.getDiff(method, quality=0.75)
        rez = len(pred_path) * 1.0
        for p in pred_path:
            rez -= p.get_quality() < 0.75
        rez = rez / len(pred_path)
        return rez

b = Benchmark()
retval = []
for vidpath in ["tests/samples/MVI_0017.AVI"]:#,"tests/samples/DSC_0004.MOV"]:
    b = Benchmark(vidpath)
    alldata = {}
    for g in [("LK",0), ("LK-SURF",0), ("SURF",10)]:
        (method, skip) = g
        print "Calculating for method " + method
        print "Taking every " + str(skip + 1) +" frame"
        print "For video " + vidpath
        cProfile.runctx("retval = b.bench_method('%s', %d)" % g, locals(), globals(), "/tmp/bench")
        print "returned " + str(retval)
        p = pstats.Stats("/tmp/bench")
        #p.strip_dirs().sort_stats("time").print_stats(10)
        data_dict = {}
        tree = CalltreeConverter(p)
        for en in tree.entries:
            if intrest_code.has_key(en.code.co_name):
                (name, repl) = intrest_code[en.code.co_name]
                if repl:
                    name = name % (method.split("-")[-1])
                data_dict[name] = en.totaltime
        alldata[method] = data_dict

print create_cvs(alldata)
