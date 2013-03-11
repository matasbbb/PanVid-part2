import cv2
from panvid.predictHelpers import *
import numpy as np

register_methods_cont = {}

class RegisterImagesCont():
    data_to_keep = 2
    def __init__(self, stream, *args):
        self._stream = stream
        self._data = []
        self._dps = []
        self._frames = [self._stream.getFrame()]
        self._sData = None
        self._method = "Not Set"
        self._methodInit(*args)

    def _methodInit(self, *args):
        return

    def _calcNext(self):
        data_to_save = None
        return data_to_save, DataPoint("None")

    def getNextDataPoint(self, skip = False):
        frame = self._stream.getFrame()
        if frame is None:
            return None
        self._frames.insert(0, frame)
        if len(self._frames) > self.data_to_keep + 2:
            self._frames.pop()

        if not skip:
            data, dp = self._calcNext()
            self._data.insert(0, data)
            self._dps.insert(0, dp)
        else:
            self._dps.insert(0, DataPoint("SKIPED"))
            self._data.insert(0, None)

        return self._dps[0]

    def getDiff(self, doneCB=None, progressCB=None):
        dp = []
        d = self.getNextDataPoint()
        while d is not None:
            if progressCB is not None:
                progressCB(self._stream.getProgress())
            dp.append(d)
            d = self.getNextDataPoint()
        if doneCB is not None:
            doneCB(dp)
        return dp

class RegisterImagesContByString(RegisterImagesCont):
    def _methodInit(self, method="LK-SURF", quality=0.5, *args):
        #Will save registerers
        if quality.__class__ == float:
            quality = [quality] * 10
        self._method = method
        methods = method.split("-")
        self._sData = []
        for m in methods:
            mstream = MockStream()
            mstream.setFrame(self._frames[0])
            reg = register_methods_cont[m](MockStream(), m)
            self._sData.append((reg, mstream, quality.pop(0)))

    def _calcNext(self):
        goodPoint = None
        found = False
        for (m,s,q) in self._sData:
            s.setFrame(self._frames[0])
            d = m.getNextDataPoint(found)
            if d.get_quality() > q and not found:
                found = True
                goodpoint = d
        if not found:
            return None, d
        else:
            return None, goodpoint

class RegisterImagesContStandart(RegisterImagesCont):
    data_to_keep = 1
    def _methodInit(self, method, *args):
        self._method = method
        self._sData = {"extractor":FeatureExtractor(method)}

    def _calcNext(self):
        if len(self._data) > 0 and self._data[0] is not None:
            (k1, d1) = self._data[0]
        else:
            (k1, d1) = self._sData["extractor"].getFeatures(self._frames[1])
        (k2, d2) = self._sData["extractor"].getFeatures(self._frames[0])
        #Match them
        if d2 is not None and d1 is not None:
            flann = MatchDesciptorsFlann(d2, self._method)
            match = flann.getPairs(d1)
            c1 = np.array([k1[e].pt for e in match[...,0]], 'float32')
            c2 = np.array([k2[e].pt for e in match[...,1]], 'float32')
        else:
            c2 = []
        #Find homography
        if (len(c2) >= 4):
            homography, mask = cv2.findHomography(c2, c1, cv2.RANSAC)
            #Asume if features matched it means they are good
            quality = min(len(c2)/100.,1.)
            quality *= mask.sum()/mask.size
            return (k2, d2), DataPoint(self._method, quality, homography)
        else:
            return (k2, d2), DataPoint(self._method)

register_methods_cont["SIFT"] = RegisterImagesContStandart
register_methods_cont["SURF"] = RegisterImagesContStandart

feature_params = dict( maxCorners = 400,
                       qualityLevel = 0.05,
                       minDistance = 35,
                       blockSize = 23,
                       useHarrisDetector = True)

lk_params = dict( winSize  = (23, 23),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005))

class RegisterImagesContLK(RegisterImagesCont):
    def _methodInit(self, *args):
        self._method = "LK"
        self._sData = lambda image: cv2.goodFeaturesToTrack(image, **feature_params)

    def getFrame(self, n):
        if self._frames[n] is not grey:
            self._frames[n] = cv2.cvtColor(self._frames[n], cv2.COLOR_BGR2GRAY)
        return self._frames[n]

    def _calcNext(self):
        for i in xrange(len(self._frames)):
            if len(self._frames[i].shape) == 3:
                self._frames[i] = cv2.cvtColor(self._frames[i], cv2.COLOR_BGR2GRAY)
        if len(self._data) > 0 and len(self._data[0]) > 1:
            p0 = self._data[0][1]
        else:
            p0 = None

        if p0 is None or len(p0) < 64:
            p0 = self._sData(self._frames[1])
        p1, st0, err = cv2.calcOpticalFlowPyrLK(self._frames[1], self._frames[0], p0, **lk_params)
        #If try back check, which reduces speed, but improves quality
        if False and doublecheck:
            p0r, st1, err = cv2.calcOpticalFlowPyrLK(self._frames[0], self._frames[1], p1, **lk_params)
            st = np.logical_and(st0.flatten(),st1.flatten())
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            back_threshold = 10.00
            #While we have enouth points reduce threshold
            #Biger threshold better, but we still want more than 16 points
            while sum(np.logical_and(d < (back_threshold/2), st)) >= 16 and back_threshold/2 > 0.01:
                back_threshold /= 2
            status = d < back_threshold
            status = np.logical_and(st,status)
        else:
            #Hack need real wa1!
            back_threshold = 0.02
            d = abs(p0).reshape(-1, 2).max(-1)
            status = True
            status = np.logical_and(st0.flatten(),status)
        p1c = p1[status].copy()
        p0c = p0[status].copy()
        #If not enougth points
        if len(p0c) < 4:
            return (None, None), DataPoint(self._method)
        else:
            homography, mask = cv2.findHomography(p1c, p0c, cv2.RANSAC)
            quality = 1.
            quality *= min(len(p0c)/100., 1.)
            quality *= 1. * mask.sum()/mask.size
            quality *= min(1., 0.02/back_threshold)
            if (len(p0c) < 40 and 1.* mask.sum()/mask.size < 0.90):
                quality *= 0.1
            marks = [back_threshold, len(p0c), 1.*mask.sum()/mask.size,
                1.*status.flatten().sum()/status.flatten().size]
            #quality *= min(1,(float(sum(mask))/len(mask)*10)
            #NOW! IF quality of last second point is 0.01.
            #And we have frame(a) (so it was sequantial.
            #Register with this frame. So we have a<->b a<->c and b<->c.
            #Test if transition is within limits of overall.
            #Set quality to better if agried.
            return (p0c, p1), DataPoint(self._method, quality, homography, marks)

register_methods_cont["LK"] = RegisterImagesContLK

