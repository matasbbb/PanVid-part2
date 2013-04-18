import cv2
from panvid.predictHelpers import *
import numpy as np

register_methods_cont = {}

class RegisterImagesCont():
    def __init__(self, stream, retain=2, *args, **kwargs):
        self._stream = stream
        self._data = []
        self._dps = []
        self._data_to_keep = retain
        self._frames = [self._stream.getFrame()]
        if self._frames[0] is None:
            print "Empty stream!"
            return
        self._sData = None
        self._method = "Not Set"
        self._methodInit(*args, **kwargs)

    def _methodInit(self, *args):
        return
    def cleanLast(self):
        self._frames.pop(0)
        if len(self._data) > 0:
            self._data.pop(0)
            self._dps.pop(0)

    def _calcNext(self):
        data_to_save = None
        return data_to_save, DataPoint("None")

    def getNextDataPoint(self, skip = False):
        frame = self._stream.getFrame()
        if frame is None:
            return None
        self._frames.insert(0, frame)
        if len(self._frames) > self._data_to_keep + 2:
            self._frames.pop()
        if len(self._data) > self._data_to_keep:
            self._data.pop()
        if not skip:
            data, dp = self._calcNext()
            self._data.insert(0, data)
            self._dps.insert(0, dp)
        else:
            self._dps.insert(0, DataPoint("SKIPED"))
            self._data.insert(0, None)

        return self._dps[0]
    def getProgress(self):
        return self._stream.getProgress()
    def getDiff(self, doneCB=None, progressCB=None):
        dp = []
        d = self.getNextDataPoint()
        while d is not None:
            if progressCB is not None:
                progressCB(self._stream.getProgress(), str(d))
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
            reg = register_methods_cont[m](stream=MockStream(),
                    retain=self._data_to_keep, method=m)
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

class RegisterImagesContJumped(RegisterImagesContByString):
    def _methodInit(self, method="LK-SURF", quality=0.5, jumpmethod="SURF",*args):
        RegisterImagesContByString._methodInit(self, method, quality)
        self.count = 0
        mstream = MockStream()
        mstream.setFrame(self._frames[0])
        reg = register_methods_cont[jumpmethod](stream=MockStream(),
                retain=self._data_to_keep, method=jumpmethod)
        self.check = (reg, mstream)

    def _calcNext(self):
        goodPoint = None
        found = False
        for (m,s,q) in self._sData:
            s.setFrame(self._frames[0])
            d = m.getNextDataPoint(found)
            #print d
            if d.get_quality() > (q * (0.98**self.count)) and not found:
                if self.count != 0:
                    print "GAPPEEED \t " + str(self.count) + " accepted \t " +str(d.get_quality())
                    self.count = 0
                found = True
                goodpoint = d


        #print (d.get_quality(), q, self.count, found)
        self.check[1].setFrame(self._frames[0])
        if found:
            self.check[0].getNextDataPoint(True)
            return None, goodpoint

        self.count += 1
        if self.count > 50:
            print "Too long gap"
            return None, None
        #Gap not acceptable, try!
        if self.count > 5 and self.count % 3 == 0:
            #Try with check method
            data = self.check[0].getNextDataPoint()
            #print data
            #Good point woohoo
            if data.get_quality() > 0.5*(0.95**self.count):
                print "WOOHOO " + str(data)
                return None, data
            else:
                #Clean, boo
                self.check[0].cleanLast()
                for (m,s,q) in self._sData:
                    m.cleanLast()
                return None, DataPoint("SKIPPED")
        else:
            #Acceptable gap, clean
            data = self.check[0].getNextDataPoint(True)
            self.check[0].cleanLast()
            for (m,s,q) in self._sData:
                m.cleanLast()

            return None, DataPoint("SKIPPED")
                
register_methods_cont["JUMPED"] = RegisterImagesContJumped


class RegisterImagesContByStringAll(RegisterImagesContByString):
    def _calcNext(self):
        points = []
        for (m,s,q) in self._sData:
            s.setFrame(self._frames[0])
            d = m.getNextDataPoint(found)
            points.append(d)
        return None, DataPoints(points)

class RegisterImagesGapedByString(RegisterImagesContByString):
    def _methodInit(self, method="LK-SURF", quality=0.5, gap=0, *args):
        RegisterImagesContByString._methodInit(self, method, quality, *args)
        self._gap = gap + 1
        self._skiped = 0

    def _calcNext(self):
        ret = None, DataPoint("Skiped")
        #print self._gap, self._skiped
        if self._skiped % self._gap == 0:
            #Only one
            ret = RegisterImagesContByString._calcNext(self)
        if self._skiped % self._gap == self._gap - 1:
            #Not intresting
            RegisterImagesContByString._calcNext(self)
        self._skiped += 1
        return ret


class RegisterImagesContStandart(RegisterImagesCont):
    _data_to_keep = 1
    def _methodInit(self, method="SURF"):
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
            quality = min(len(c2)/250.,1.)
            quality *= mask.sum()/mask.size
            return (k2, d2), DataPoint(self._method, self._frames[0].shape, quality, homography, [len(c2), mask.sum()/mask.size])
        else:
            return (k2, d2), DataPoint(self._method)

register_methods_cont["SIFT"] = RegisterImagesContStandart
register_methods_cont["SURF"] = RegisterImagesContStandart

feature_params = dict( maxCorners = 400,
                       qualityLevel = 0.05,
                       minDistance = 33,
                       blockSize = 23,
                       useHarrisDetector = False)

lk_params = dict( winSize  = (23, 23),
                  maxLevel = 3,
                  minEigThreshold = 0.0001,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005))

class RegisterImagesContLK(RegisterImagesCont):
    def _methodInit(self, method="LK", *args):
        self._method = "LK"
        self._maxF = self._frames[0].shape[0]/feature_params["minDistance"] * \
                     self._frames[0].shape[1]/feature_params["minDistance"] / 10
        self._maxF = None
        
        self._sData = lambda image: cv2.goodFeaturesToTrack(image, **feature_params)
        self._points = self._frames[0].shape[0]*self._frames[0].shape[1]*0.0001

    def predictCheap(self, pre_points, frame0, frame1, prevHomo=None):
        p1=None
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
        if prevHomo is not None and False:
            p1=cv2.perspectiveTransform(pre_points, prevHomo)
            flags+=cv2.OPTFLOW_USE_INITIAL_FLOW
        p0 = pre_points
        if self._frames[frame1].shape != self._frames[frame0].shape:
            return p0, [], DataPoint(self._method, marks=marks)
        p1, st0, err = cv2.calcOpticalFlowPyrLK(self._frames[frame1], self._frames[frame0], p0, p1, flags=flags, **lk_params)
        self._count += 1
        d = abs(p0).reshape(-1, 2).max(-1)
        status = True
        status = np.logical_and(st0.flatten(),status)
        p1c = p1[status].copy()
        p0c = p0[status].copy()
        marks = [len(p0c), 0,
           1.*status.flatten().sum()/status.flatten().size]
        
        if len(p1c) < 4:
            return p0c, p1c, DataPoint(self._method, marks=marks)
        homography, mask = cv2.findHomography(p1c, p0c, cv2.RANSAC)
        quality = 1.
        quality *= min(len(p0c)/(self._points), 1.)
        quality *= 1. * mask.sum()/mask.size
        #if (len(p0c) < 40 and 1.* mask.sum()/mask.size < 0.90):
        #   quality *= 0.1
        marks[1] = 1.*mask.sum()/mask.size
 
        return p0c, p1c, DataPoint(self._method, self._frames[0].shape, quality, homography, marks)
    #     [(old_points, new_points).(old_points, new_points)]
    # frame[0]<-data[0][1]<-frame[1]<-data[0][0]<(posible =) - data[1][1]-frame[2]<-data[1][0]  
    def getFromPoints(self, f, new=True):
        points = None
        if len(self._data) > f and len(self._data[f]) > 1:
             points = self._data[f][new]
        return points

    def calcPoints(self, f, points=None):
        #more points
        if points is None:
            points = None
        #Clever clean up?
        points = self._sData(self._frames[f])
        if points is None:
            print "WTF!"
            return []
        if self._maxF is None:
            self._maxF = len(points)
        if self._maxF < len(points):
            self._maxF = len(points)
        if self._maxF > len(points):
            self._maxF = self._maxF * 0.8 + len(points) * 0.2
        return points

    def checkGray(self):
        for i in xrange(len(self._frames)):
            if len(self._frames[i].shape) == 3:
                self._frames[i] = cv2.cvtColor(self._frames[i], cv2.COLOR_BGR2GRAY)

    def _calcNext(self):
        self._count = 0
        self.checkGray()
        
        #Try get frames from old frame.
        from_points = self.getFromPoints(0)
        
        #If none or not enouth calculate from this frame
        if from_points is None or len(from_points) < self._maxF*3.5/5.:
            from_points = self.calcPoints(0, from_points)
        
        if len(from_points) < 4:
            #Cant do anything better!
            return (from_points, []), DataPoint(self._method)
        
        homo = None
        if len(self._dps) > 0 and self._dps[0].get_quality > 0.9:
            homo = self._dps[0].get_homo()
            
        #Predict simply
        prev_croped, new_points_1, dp0_1 = self.predictCheap(from_points,0,1,homo)


        dp0_1._marks.append(dp0_1._quality)
        #Doublecheck?
        other_dps = [dp0_1]
        back, num,dp = 0,0,0
        if dp0_1._quality < 0.95:
            #double is better!
            back, num, dp, newdp0_1 = self.double(prev_croped, new_points_1)
            if newdp0_1._quality > 0.95:
                newdp0_1._marks = dp0_1._marks
                dp0_1 = newdp0_1
                other_dps.insert(0,dp0_1)
            else:
                other_dps.append(newdp0_1)
            dp0_1._marks.append(newdp0_1._quality)
        else:
            dp0_1._marks.append(0)

            #dst = copydp0_1.get_distance(dp0_1)
            #dp0_1._marks.append(dst, num)
            #if dst is not None and dst != 0:
            #    dp0_1._quality *= min(1,30/dst)
                #dp0_1._quality *= min(1,2*num/len(prev_croped))
            #    dp0_1._quality = double_dp._quality
        dp0_1._marks.append(back)
        dp0_1._marks.append(num)
        dp0_1._marks.append(dp)
       
        #Check with last frame, it will update it
        dp0_1 = self.checkback(other_dps, quality=0.95)
        
        #print str(self._count) + "     " + str(dp0_1._quality )
        return (from_points, new_points_1), dp0_1
        
    def checkback(self, dps, quality=0.98, distance=50):
        dist = 10000
        #If bad and we have last homo check against it!
        dp0_1 = dps.pop(0)
        if dp0_1.get_quality() < quality:
            from_points_2 = self.getFromPoints(1)
            if from_points_2 is not None and \
                    len(from_points_2) > 4 and  \
                    self._dps[0].get_quality() > 0:
                prev_croped, new_points_2, dp0_2 = self.predictCheap(from_points_2,0,2)
                dp1_2 = self._dps[0]
                dist = dp0_2.get_distance(dp1_2*dp0_1)
                #Get lower if have second
                for dpn in dps:
                    if dpn is not None:
                        ndist = dp0_2.get_distance(dp1_2*dpn)
                        if ndist is not None and ndist < dist:
                            dist = ndist
                            #print "Happended"
                            dp0_1._homo = dpn._homo 

                if dist is None:
                    dist = 10000
                if dist < distance:
                    dp0_1._quality += dp0_2._quality + dp1_2._quality
                    dp0_1._quality = min(dp0_1._quality/2., 1.)
                    dp0_1._quality *= min(1.*len(from_points_2)/self._maxF,1.)
                    dp0_1._quality *= min(((distance-dist)*2/distance),1.)
                else:
                    #we need to decrease quality
                    dp0_1._quality *= distance/dist
        dp0_1._marks.append(dist)
        return dp0_1

    def double(self, p0, p1, back_threshold=10., points=8, mini=0.2):
        #If try back check, which reduces speed, but improves quality
        p0r, st1, err = cv2.calcOpticalFlowPyrLK(self._frames[0], self._frames[1], p1, **lk_params)
        self._count += 1
        if st1 is None:
            return 0, back_threshold, 0, DataPoint(self._method)    
        st = st1.flatten()
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        #While we have enouth points reduce threshold
        #Biger threshold better, but we still want more than 16 points
        with_01 = np.logical_and(d < mini, st).sum()
        while sum(np.logical_and(d < (back_threshold/2), st)) >= points and back_threshold/2 > 0.1:
            back_threshold /= 1.5
        status = d < back_threshold
        status = np.logical_and(st,status)
        p1c = p1[status].copy()
        p0c = p0[status].copy()
        if len(p0c) >  4:
            #Different direction!
            homography, mask = cv2.findHomography(p1c, p0c, cv2.RANSAC)
            quality = 0.97 * mask.sum()/mask.size
            quality *= min(1.,with_01/(self._points))
            return with_01, back_threshold, 1.*mask.sum()/mask.size, DataPoint(self._method, self._frames[0].shape, quality, homography)
        else:
            return with_01, back_threshold, 0, DataPoint(self._method)    


register_methods_cont["LK"] = RegisterImagesContLK

