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
import os
import resource
from functools import partial
intrest_code = \
    {'<cv2.findHomography>':("RANSAC", 0),
     '<cv2.goodFeaturesToTrack>':("LK Feature", 0),
     '<cv2.calcOpticalFlowPyrLK>':("LK Compute", 0),
     "<method 'compute' of 'cv2.DescriptorExtractor' objects>":("%s Compute", 1),
     "<method 'detect' of 'cv2.FeatureDetector' objects>":("%s Feature", 1)
    }
def write_to_file(data, filename, option="w"):
    f = open(filename, option)
    f.write(str(data))
    f.close()

def to_file(data, filename, option="w"):
    f = open(filename, option)
    for d in data:
        f.write(str(d.get_quality()) + "\n")
    f.close()

def using(point):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return point + " " + str(usage[2]) + "\n"

def dirSamples(url):
    files = os.walk(url).next()
    l = []
    for f in files[2]:
        st = cv2.VideoCapture(url + "/" + f)
        if st.isOpened():
            fr = st.read()
            if fr[0]:
                size = fr[1].shape[0] * fr[1].shape[1] * 0.000001
                l.append((size, url+"/" +f))
    return l

class Benchmark(object):
    vidsamples = [
             #     (480*640*0.000001,"tests/samples/MVI_0017.AVI"),
             #     (720*1280*0.000001, "tests/samples/DSC_0004.MOV"),
             #     (1920*1080*0.000001,"tests/samples/DSC_0011.MOV"),
                ]
    vidsampledirs = [
            "tests/samples/moresample"
            ]
    images = [
              #"/home/matas/part2/tests/samples/IMG_7955.JPG",
              #"/home/matas/part2/tests/samples/IMG_8686.JPG",
              #"/home/matas/part2/tests/samples/INT_NOT.jpg",
              ]
    #framesizes = [(500,4000),(707,2828),(1000,2000),(1069,1871),(1155,1732),(1265,1581),(1414,1414)]
    #framesizes = [(500,600),(900,1000),(1000,2000)]
    framesizes = [(480,640),(720,1280),(1080,1920),(1500,2500),(1500,3000),(2500,3000)]
    framesizes = framesizes[0:5]
    methods = [("LK",0),("SURF", 0),("SIFT", 0)]
    #methods = [("LK",0), ("SURF",0)]
    #methods = [methods[0]] * 1
    methods = [("LK",0)]
    gen = []


    def __init__(self, start, framenumber, seq=True, real_compare=None, progress=True):
        self.datadir = "/media/matas/gdrive/new/"
        self.real_compare=real_compare
        self.start = start
        self.bound = 0
        if framenumber != 0:
            self.bound = start + framenumber
        self.real_now = 0
        self.synth_now = 0
        self.homos = {}
        self.fr = framenumber
        if self.fr == 0:
            self.fr = 1
        self.seq = seq
        self.prevHomos = PredictSave()
        for d in self.vidsampledirs:
            self.vidsamples += dirSamples(d)
        self.vidsamples = list(set(self.vidsamples))
        self.proxy = lambda stream: StreamProxyResize(stream, sizef=(0.5,0.5))
        self.proxy = lambda stream: StreamProxyCrop(stream, size=(720,1280))
        self.proxy = None
        if self.proxy is not None or True:
            nvid = []
            for size, url in self.vidsamples:
                if size > 0.5:
                    nvid.append((size, url))
            self.vidsamples = nvid

        if progress:
            self._progressCB = lambda *args: print (args)
        else:
            self._progressCB = None
        #self._progressCB = lambda *args: write_to_file(using(args[0]),"/media/matas/gdrive/new/memory.txt", "a")

    def getImageStr(self, imageID):
        return self.images[imageID].split("/")[-1].split(".")[0]

    def getVideoStr(self, vidID):
        return ".".join(self.vidsamples[vidID][1].split("/")[-1].split(".")[:-1])


    def getFrameStr(self, frameID):
        return str(self.framesizes[frameID])

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
    
    def genVideos(self, framesizeID, imageID):
        filename = "/tmp/" + "VID" + self.getFrameStr(framesizeID) + \
                self.getImageStr(imageID) + ".avi"
        frame_size = self.framesizes[framesizeID]
        if filename in self.gen or os.path.exists(filename):
            return filename
        img = cv2.imread(self.images[imageID])
        p_w = (img.shape[1]-frame_size[1])/50 #15 pixels between frames
        if self.bound > p_w:
            print ("we need more points..." + str(img.shape)+ str(p_w))
            #return
        gen = PathGenerator(p_w, img, None, self.framesizes[framesizeID])
        path = gen.getSweapPath(2000, False)
        #crop to 100
        path = path[:110]
        filt = PathFilters(path)
        filt.applySpeedChange(speed=60).applyShake(shake=8)
        path = filt.fixPath(frame_size, (img.shape[0], img.shape[1])).getPath()
        print (len(path))
        vidgen = VideoSimpleGenerator(path, img)
        vidfilt=VideoEffects().noise()
        vidfilt=None
        vidgen.save(filename, frame_size, fps=30, filt=vidfilt)
        self.last_path = []
        lt = path[0]
        for t in path[1:]:
            self.last_path.append((round(t[0])-round(lt[0]),
                                   round(t[1])-round(lt[1])))
            lt = t
        homos = []
        for p in self.last_path:
            homos.append(DataPoint("synth", 1,
                np.matrix([[1,0,p[1]],[0,1,p[0]],[0,0,1]])))
        self.homos[(framesizeID,imageID)] = homos
        self.gen.append(filename)
        return filename

    def real_data_bench(self, methodID, vidsampleID):
        size, vidpath = self.vidsamples[vidsampleID]
        method, skip = self.methods[methodID]
        print ("Real method:" + method + ", sample" + str(self.getVideoStr(vidsampleID)))
        stream = VideoInputAdvanced(vidpath, self.bound, skip, self.start)
        if self.proxy is not None:
            stream = self.proxy(stream)
        register = RegisterImagesContByString(stream, self.seq*2, method)
        #HaACK!
        #register = RegisterImagesGapedByString(stream, 3, method, 0.5, 10)
        pred_path = register.getDiff(progressCB=self._progressCB)
        if self.proxy is not None:
            pred_path = stream.modifyDataPoints(pred_path)
        if self.real_compare is not None:
            paths = []
            for methcomp in self.real_compare:
                #TODO skip if skip!
                ident = (methcomp, vidpath, "20m")
                data = self.prevHomos.getData(ident)
                if data is None:
                    print ("No data for " + str(ident))
                    #generate for all video!
                    stream = VideoInputAdvanced(vidpath, skip=skip, bound=20*30)
                    register = RegisterImagesContByString(stream, 3, methcomp)
                    data = register.getDiff(progressCB=self._progressCB)
                    self.prevHomos.setData(ident, data)
                #Crop data
                good_path = data[self.start:]
                paths.append(good_path)
            self.compare(paths, pred_path,
                    method+"_to_"+"-".join(self.real_compare) + "_" + \
                                self.getVideoStr(vidsampleID)+
                                str(self.proxy is None)+"_r.txt")

        return pred_path

    def compare(self, paths, npath, filename):
        f = open(self.datadir+filename,"w")
        fun = partial(zip)
        for p in paths:
            fun = partial(fun, p)
        for (ps,dp) in zip(fun(), npath):
            np.set_printoptions(precision=3,suppress=True)
            s = ""
            for p in ps: 
                dist = dp.get_distance(p)
                p._cor = dp._cor
                p._shape = dp._shape
                if dist == None:
                    dist = 10000
                s += str(dist) + " "
            if len(ps) == 2:
                mid = ps[0].get_middle(ps[1])
                dist = dp.get_distance(mid)
                if dist == None:
                    dist = 10000
                s += str(dist) + " "
                dist = ps[0].get_distance(ps[1])
                if dist == None:
                    dist = 10000
                s += str(dist) + " "

            s += str(dp.get_quality())
            for m in dp.get_marks():
                s += " " + str(m)
            
            for m in p.get_marks():
                s += " " + str(m)

            s =s + " \n"
            #HaACK!
            #if dist != 10000:
            #if dist > 10:
            f.write(s)
        f.close()
        return

    def synth_data_bench(self, methodID, framesizeID, imageID):
        vidpath = self.genVideos(framesizeID, imageID)
        method, skip = self.methods[methodID]
        print ("Syntetic with method:" + method + ", frame" + str(framesizeID) + ",for image:" + str(imageID))
        stream = VideoInputAdvanced(vidpath, self.bound, skip, self.start)
        register = RegisterImagesContByString(stream, self.seq*2, method)
        pred_path = register.getDiff(progressCB=self._progressCB)
        if len(pred_path) < self.bound-self.start:
            print ("Not all points!? " + str(len(pred_path)))
        if self.homos.has_key((framesizeID,imageID)):
            self.compare([self.homos[(framesizeID,imageID)]], pred_path,
                    method+"_to_"+str(self.framesizes[framesizeID])+"_"+self.getImageStr(imageID)+"_s.txt" )
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
                    s += sep + str(data[0][methID][frameID][imgID])
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
                s += sep + str(data[1][methID][sampleID])
            s += "\n"
        return s

#b = Benchmark(0,100,True,["SURF"])
b = Benchmark(0,600,True,["SURF", "SIFT"], progress=False)
#b = Benchmark(0,400,True, progress=False)
totaltime = {0:{},1:{}}

cProfile.runctx("rez = b.next_bench()", locals(), globals(), "/tmp/bench")
while rez:
    p = pstats.Stats("/tmp/bench")
    p.strip_dirs().sort_stats("time").print_stats(5)
    data_dict = {}
    tree = CalltreeConverter(p)
    (real, rez) = rez
    if real:
        methodID, vidsampleID, rez = rez
    else:
        methodID, framesizeID, imageID, rez = rez

    if real and False:
        stream = VideoInputAdvanced(b.vidsamples[vidsampleID][1], bound=b.bound, start=b.start)
        stream = StreamProxyBorder(stream,borderWidth=6, borderColorEnd=(255,0,0))
        #blend = BlendOverlay2D(stream.getClone())
        #blend.blendNextN(rez, False, False)
        #cv2.imwrite(b.datadir+"/images/"+ str(methodID) + b.getVideoStr(vidsampleID)+ "2d.jpg",blend.getPano())
        blend = BlendOverlayHomo(stream.getClone())
        blend.blendNextN(rez, False, False)
        cv2.imwrite(b.datadir+"/images/"+ str(methodID) + b.getVideoStr(vidsampleID) + "homo.jpg",blend.getPano())



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
    skip = b.methods[methodID][1] + 1
    if real:
        totaltime[real][methodID][vidsampleID] = skip* t / (len(rez)+1)
    else:
        totaltime[real][methodID][framesizeID][imageID] = skip * t / (len(rez) +1)

    #alldata[methodID] = data_dict

    #othertype[[methodID][sizeID] = t
    #print (create_cvs(alldata))
    if real:
        filename = b.datadir + "points" + str(b.vidsamples[vidsampleID][0]) + ".cvs"
    else:
        filename = b.datadir + "points" + str(b.framesizes[framesizeID]) +  ".cvs"
    to_file(rez, filename)
    cProfile.runctx("rez = b.next_bench()", locals(), globals(), "/tmp/bench")

s = b.nice_times(totaltime)
filename = b.datadir + "time_"
filename += str(b.fr) + " " + str(b.start)
if not b.seq:
    filename += "not"
filename += "seq_" + "_".join(map(lambda x: x[0], b.methods))
f = open(filename + ".txt", "w")
f.write(s)
f.close()
print(s)
