import math
import cv2
import numpy
from random import *


class PathFilters(object):
    def __init__(self, path):
        self._path = path

    def applyShake(self, shake=3, factor=0.9, ease=0.3):
        """ Applies sort of shake on camera. Easies througth frames"""
        xrand = 0
        yrand = 0
        newpath = []
        for point in self._path:
            xrand = xrand * ease + gauss(0, factor) * shake * (1 - ease)
            yrand = yrand * ease + gauss(0, factor) * shake * (1 - ease)
            newpath.append((point[0] + xrand, point[1] + yrand))
        self._path = newpath
        return self

    def applySpeedChange(self, speed=10, factor=0.5, ease=0.7, retval=0.01):
        """ Speeds up or slows down movement"""
        xsp = 0
        ysp = 0
        xdiff = 0
        ydiff = 0
        lastpoint = self._path[0]
        newpath = [lastpoint]
        for point in self._path[1:]:
            xsp = xsp * ease + gauss(-1 * retval * xdiff, factor) * speed * (1 - ease)
            ysp = ysp * ease + gauss(-1 * retval * ydiff, factor) * speed * (1 - ease)
            xdiff += xsp
            ydiff += ysp
            newpath.append((point[0] + xdiff, point[1] + ydiff))
        self._path = newpath
        return self


    def getPath(self):
        return self._path;


class PathGenerator(object):
    """Generates path for camera movement"""
    def __init__(self, framecount=1, image=None,
                 bounds=(1280, 720), vsize=(1280, 720)):
        self._bounds = bounds
        if image is not None:
            self._bounds = (image.shape[0], image.shape[1])
        if self._bounds[0] < vsize[0] or self._bounds[1] < vsize[1]:
            raise Exception("Image is smaller then desired video")
        self._vsize = vsize
        self._framecount = framecount

    def getSweapPath(self, pos, vert=True):
        """Generates simple sweap at indicated position"""
        if not vert:
            return self.getPath([(pos, self._vsize[0] / 2),
                                 (pos, self._bounds[0] - self._vsize[0] / 2)])
        else:
            return self.getPath([(self._vsize[1] / 2, pos),
                                 (self._bounds[1] - self._vsize[1] / 2, pos)])

    def getPath(self, points):
        """Generates array of points, which will corespond to each frame,
            Could adjust frame number"""
        #At start calculate aproximate path length.
        pathlen = 0.
        lastpoint = points[0]
        for point in points[1:]:
            pathlen += math.sqrt((point[0] - lastpoint[0])**2 +
                                 (point[1] - lastpoint[1])**2)
            lastpoint = point
        speed = pathlen / (self._framecount - 1)
        framepoints = []
        lastpoint = points[0]
        for point in points[1:]:
            xdiff = point[0] - lastpoint[0]
            ydiff = point[1] - lastpoint[1]
            if xdiff != 0:
                xlam = speed / ((1 + (ydiff / xdiff)) ** 2)
            else:
                xlam = 0
            if ydiff != 0:
                ylam = speed / ((1 + (xdiff / ydiff)) ** 2)
            else:
                ylam = 0
            #While point aproaching net point.
            while ((round(lastpoint[0], 2) < point[0]) == (xlam > 0)) and \
                  ((round(lastpoint[1], 2) < point[1]) == (ylam > 0)):
                framepoints.append((lastpoint[0], lastpoint[1]))
                lastpoint = (lastpoint[0] + xlam, lastpoint[1] + ylam)
            lastpoint = point
            framepoints.append(lastpoint)
        return framepoints


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


    def noise(self, noise_type=cv2.cv.CV_RAND_UNI, parm1=0, parm2=256):
        self._toUse.append(lambda frame: self._noise(noise_type, frame, parm1,
                parm2))

    def _noise(self, noise_type, frame, parm1, parm2):
        rand = cv2.cv.CreateImage((frame.shape[0],frame.shape[1]), 8, frame.shape[2])
        cv2.cv.RandArr(cv2.cv.RNG(0), rand, cv2.cv.CV_RAND_NORMAL, parm1, parm2)
        frame = cv2.cv.fromarray(frame)
        cv2.cv.Add(rand, frame, frame)
        return numpy.asarray(frame)

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


