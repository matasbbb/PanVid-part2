from __future__ import print_function
from panvid.generateVideo import *
from panvid.generatePath import *
from panvid.predict import *
from panvid.input import *
from panvid.inputProxy import *
from panvid.pyprof2calltree import *
from panvid.blend import *
from panvid.predictSave import *
import numpy as np
import cv2
import cProfile
import pstats
import math
intrest_code = \
    {'<cv2.findHomography>':("RANSAC", 0),
     '<cv2.goodFeaturesToTrack>':("LK Feature", 0),
     '<cv2.calcOpticalFlowPyrLK>':("LK Compute", 0),
     "<method 'compute' of 'cv2.DescriptorExtractor' objects>":("%s Compute", 1),
     "<method 'detect' of 'cv2.FeatureDetector' objects>":("%s Feature", 1)
    }

def to_file(data, filename):
    f = open(filename,"w")
    for d in data:
        f.write(str(d.get_quality()) + "\n")
    f.close()

class Benchmark(object):
    vidsamples = [
                 # (0.3,"tests/samples/MVI_0017.AVI"),
                 # (0.9, "tests/samples/DSC_0004.MOV"),
                  (2,"tests/samples/DSC_0011.MOV"),
                ]
    images = [
              #"/home/matas/part2/tests/samples/IMG_7955.JPG",
              #"/home/matas/part2/tests/samples/IMG_8686.JPG",
              ]
    framesizes = [(500,600),(900,1000),(1000,2000)]
    methods = [("LK",0),("SURF", 0),("SIFT", 0)]
    methods = [("LK",0), ("SURF",0)]
    methods = [("SURF",0)]
    gen = []


    def __init__(self, start, framenumber, seq=True, real_compare=None):
        self.real_compare=real_compare
        self.start = start
        self.bound = 0
        if framenumber != 0:
            self.bound = start + framenumber
        self.real_now = 0
        self.synth_now = 0
        self.fr = framenumber
        if self.fr == 0:
            self.fr = 1
        self.fmask = None
        if not seq:
            self.fmask = [True,False] * framenumber
            self.bound += framenumber
        self.seq = seq
        self.prevHomos = PredictSave()
        self.proxy = lambda stream: StreamProxyResize(stream, sizef=(0.5,0.5))
        self.proxy = lambda stream: StreamProxyCrop(stream, size=(1280,720))

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
        return ret_string

    def genVideos(self, frame_size, imgpath):
        filename = "/tmp/" + "VID" + str(frame_size) + str(self.images.index(imgpath)) +".avi"
        if filename in self.gen:
            return filename
        img = cv2.imread(imgpath)
        gen = PathGenerator(200, img, None, frame_size)
        path = gen.getSweapPath(2000, False)
        spshpath = PathFilters(path).applySpeedChange(speed=30).applyShake().fixPath(frame_size, (img.shape[0], img.shape[1])).getPath()
        if self.bound != 0:
            spshpath = spshpath[:self.bound+1]
        vidgen = VideoSimpleGenerator(spshpath, img)
        filt = VideoEffects()
        filt.noise()
        vidgen.save(filename, frame_size, fps=30, filt=filt)
        self.last_path = []
        lt = spshpath[0]
        for t in spshpath[1:]:
            self.last_path.append((round(t[0])-round(lt[0]),round(t[1])-round(lt[1])))
            lt = t

        self.homos = []
        for p in self.last_path:
            self.homos.append(DataPoint("synth",1,np.matrix([[1,0,p[1]],[0,1,p[0]],[0,0,1]])))

        self.gen.append(filename)
        return filename

    def real_data_bench(self, methodID, vidsampleID):
        size, vidpath = self.vidsamples[vidsampleID]
        method, skip = self.methods[methodID]
        stream = VideoInputAdvanced(vidpath, skip=skip, start=self.start, bound=self.bound)
        if self.proxy is not None:
            stream = self.proxy(stream)
        register = RegisterImagesDetect(stream)
        progressCB = lambda *args: print (args)
        pred_path = register.getDiff(method, quality=0.80, progressCB=progressCB, fmask=self.fmask)
        if self.proxy is not None:
            pred_path = stream.modifyDataPoints(pred_path)
        if self.real_compare is not None:
            for methcomp in self.real_compare:
                register = RegisterImagesDetect(stream)
                ident = (methcomp,skip, vidpath, self.seq)
                data = self.prevHomos.getData(ident)
                if data is None:
                    #generate for all video!
                    stream = VideoInputAdvanced(vidpath, skip=skip)
                    register = RegisterImagesDetect(stream)
                    data = register.getDiff(methcomp, quality=0.80, progressCB=progressCB, fmask=self.fmask)
                    self.prevHomos.setData(ident, data)
            #Crop data
                good_path = data[self.start:]
                self.compare(good_path, pred_path, method+"to"+methcomp+str(vidsampleID)+"r.txt" )

        return pred_path

    def compare(self, path, npath, filename):
        print (filename)
        f = open("/tmp/"+filename,"w")
        for (p,dp) in zip(path, npath):
            cor = np.array([[0,0],[0,1000],[1000,0],[1000,1000]],dtype='float32')
            cor = np.array([cor])
            if dp.get_homo() is None:
                continue
            np.set_printoptions(precision=3,suppress=True)
            des = cv2.perspectiveTransform(cor, dp.get_homo())
            desn = cv2.perspectiveTransform(cor, p.get_homo())
            sq = 0.
            dist = 0.
            for p1,p2 in zip(des[0], desn[0]):
                dist += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                sq += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            s = str(dp.get_quality())
            for m in dp.get_marks():
                s += " " + str(m)
            s += " " + str(dist) + " " + str(sq) +" \n"
            f.write(s)
        f.close()
        return

    def synth_data_bench(self, methodID, framesizeID, imageID):
        vidpath = self.genVideos(self.framesizes[framesizeID], self.images[imageID])
        method, skip = self.methods[methodID]
        stream = VideoInputAdvanced(vidpath, skip=skip, start=self.start, bound=self.bound)
        register = RegisterImagesDetect(stream)
        progressCB = lambda *args: print (args)
        pred_path = register.getDiff(method, quality=0.80, progressCB=progressCB, fmask=self.fmask)
        self.compare(self.homos, pred_path, str(methodID)+" "+str(framesizeID)+" "+str(imageID)+"s.txt" )
        return pred_path

    def next_real_data_bench(self):
        methodID = self.real_now % len(self.methods)
        vidsampleID = self.real_now / len(self.methods)
        if vidsampleID >= len(self.vidsamples):
            return False;
        self.real_now += 1
        rez = self.real_data_bench(methodID, vidsampleID)
        return (methodID, vidsampleID, rez)

    def next_synthetic_data_bench(self):
        methodID = self.synth_now % len(self.methods)
        framesizeID = self.synth_now / len(self.methods) % len(self.framesizes)
        imageID = self.synth_now / len(self.methods) / len(self.framesizes)
        if imageID >= len(self.images):
            return False
        self.synth_now += 1
        rez = self.synth_data_bench(methodID, framesizeID, imageID)
        return (methodID, framesizeID, imageID, rez)

    def next_bench(self):
        rez = self.next_real_data_bench()
        if rez:
            return (1, rez)
        else:
            rez = self.next_synthetic_data_bench()
            if rez:
                return (0, rez)
            else:
                return False

    def nice_times(self, data, sep=" "):
        s = ""
        for imgID in xrange(len(self.images)):
            s +=  "data for " + self.images[imgID] + "\n"
            #Header
            s += "Mpx"
            for methID in xrange(len(self.methods)):
                s += sep + self.methods[methID][0]
            s += "\n"
            #Data
            for frameID in xrange(len(self.framesizes)):
                s += str(0.000001 * self.framesizes[frameID][0] * self.framesizes[frameID][1])
                for methID in xrange(len(self.methods)):
                    s += sep + str(data[0][methID][frameID][imgID]/self.fr*(self.methods[methID][1]+1))
                s += "\n"

        s += "\n Real life data \n"
        #Header
        s += "Mpx"
        for methID in xrange(len(self.methods)):
            s += sep + self.methods[methID][0]
        s += "\n"

        for sampleID in xrange(len(self.vidsamples)):
            s += str(self.vidsamples[sampleID][0])
            for methID in xrange(len(self.methods)):
                s += sep + str(data[1][methID][sampleID]/self.fr*(self.methods[methID][1]+1))
            s += "\n"
        return s

b = Benchmark(0,100,True,["SURF"])
totaltime = {0:{},1:{}}

cProfile.runctx("rez = b.next_bench()", locals(), globals(), "/tmp/bench")
while rez:
    if False:
        stream = VideoInputAdvanced(vidpath,skip=skip)
        blend = BlendOverlay2D(stream)
        blend.blendNextN(retval, True, False)
        cv2.imwrite("/tmp/"+ method + str(i)+ ".jpg",blend.getPano())

    p = pstats.Stats("/tmp/bench")
    #p.strip_dirs().sort_stats("time").print_stats(10)
    data_dict = {}
    tree = CalltreeConverter(p)
    (real, rez) = rez
    if real:
        methodID, vidsampleID, rez = rez
    else:
        methodID, framesizeID, imageID, rez = rez

    if not totaltime[real].has_key(methodID):
        totaltime[real][methodID] = {}
    if not real and not totaltime[real][methodID].has_key(framesizeID):
        totaltime[real][methodID][framesizeID] = {}

    t = 0.
    for en in tree.entries:
        if intrest_code.has_key(en.code.co_name):
            #(name, repl) = intrest_code[en.code.co_name]
            #if repl:
            #    name = name % (method.split("-")[-1])
    #        data_dict[name] = en.totaltime
            t += en.totaltime
    if real:
        totaltime[real][methodID][vidsampleID] = t
    else:
        totaltime[real][methodID][framesizeID][imageID] = t

    #alldata[methodID] = data_dict

    #othertype[[methodID][sizeID] = t
    #print (create_cvs(alldata))
    if real:
        filename = "/tmp/points" + str(b.vidsamples[vidsampleID][0]) + ".cvs"
    else:
        filename = "/tmp/points" + str(b.framesizes[framesizeID]) +  ".cvs"
    to_file(rez, filename)
    cProfile.runctx("rez = b.next_bench()", locals(), globals(), "/tmp/bench")

s = b.nice_times(totaltime)
f = open("/tmp/rez"+str(b.fr) + " " + str(b.start) + "seq_5KLK.csv", "w")
f.write(s)
f.close()
print(s)
