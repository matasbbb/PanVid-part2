import math
import cv2
import numpy
from random import *
import time

class VideoEffects(object):
    def __init__(self):
        self._toUse = []
        self._required_size = (1,1)

    def getMult(self):
        return self._required_size

    def crop(self, frame_size, pos=None, mult = (1,1), end=True):
        self._toUse.insert(end * -1,
                lambda frame: self._crop(frame_size, mult, pos, frame))

    def _crop(self, frame_size, mult, pos, frame):
        frame_size = (mult[0] * frame_size[0], frame_size[1] * mult[1])
        if pos is None:
            pos = (frame.shape[0] / 2, frame.shape[1] / 2)
        xleft = round(pos[0] - frame_size[0] / 2)
        ytop = round(pos[1] - frame_size[1] / 2)
        #Prefer bigger frame
        xright = xleft + math.ceil(frame_size[0])
        ydown = ytop + math.ceil(frame_size[1])
        frame = frame[xleft:xright, ytop:ydown].copy()
        return frame


    def noise(self, noise_type=cv2.cv.CV_RAND_UNI, parm1=0, parm2=50):
        self._toUse.append(lambda frame: self._noise(noise_type, frame, parm1, parm2))

    def _noise(self, noise_type, frame, parm1, parm2):
        rand = cv2.cv.CreateImage((frame.shape[0], frame.shape[1]), 8, frame.shape[2])
        cv2.cv.RandArr(cv2.cv.RNG(round(time.time())), rand, cv2.cv.CV_RAND_NORMAL, parm1, parm2)
        frame = cv2.cv.fromarray(frame)
        cv2.cv.Add(rand, frame, frame)
        retframe = numpy.asarray(frame)
        return retframe

    def apply(self, frame):
        cp = frame
        for fun in self._toUse:
            cp = fun(cp)
        return cp

class VideoSimpleGenerator(object):
    def __init__(self, path, image):
        self._path = path
        self._image = image

    def save(self, url, frame_size, fourcc=cv2.cv.FOURCC("P","I","M","1"),
             fps=30, filt=None):
        writer = cv2.VideoWriter(url, fourcc, fps, frame_size)
        #Always post crop to wanted size
        if filt is None:
            filt = VideoEffects()
            filt.crop(frame_size)
        for pos in self._path:
            #Crop at end
            #filt.crop(frame_size)
            #Crop at start to work with smaller image
            prefilt = VideoEffects()
            prefilt.crop(frame_size, pos, filt.getMult())
            frame = prefilt.apply(self._image)
            frame = filt.apply(frame)
            #x = round(frame[0] - frame_size[0]/2)
            #y = round(frame[1] - frame_size[1]/2)
            #frame = self._image[x:x+frame_size[0],y:y+frame_size[1]].copy()
            #if filt is not None:
            #    frame = filt.apply(frame)
            writer.write(frame)
        writer.release()


