import cv2
import numpy as np

class MockStream():
    frames = []
    def setFrame(self, frame):
        self.frames.append(frame)
    def getFrame(self):
        return self.frames.pop(0)

class DataPoint():
    def __init__(self, method, shape=(1000,1000), quality=0, homo=None, marks=[]):
        self._method = method
        self._quality = quality
        self._homo = homo
        self._marks = marks
        self._shape = shape
        cor = np.array([[0,0],[0,shape[1]],[shape[0],0],[shape[0],shape[1]]],dtype='float32')
        self._cor = np.array([cor])
        if homo is None:
            self._quality = 0

    def get_homo(self):
        return self._homo

    def get_naive2D(self):
        if self._homo is not None:
            return (self._homo[1][2],self._homo[0][2])
        else:
            return (0,0)

    def get_better2D(self):
        if self._homo is not None:
            mid = [self._shape[1]/2, self._shape[0]/2]
            amid = np.array([mid], dtype='float32')
            nmid = cv2.perspectiveTransform(np.array([amid]), self._homo)[0][0]
            return (nmid[1]-mid[1],nmid[0]-mid[0])
        else:
            return (0,0)

    def get_quality(self):
        return self._quality
    def get_marks(self):
        return self._marks
    def get_distance(self, datapoint):
        if self._homo is not None  and datapoint.get_homo() is not None:
            des = cv2.perspectiveTransform(self._cor, self._homo)
            desn = cv2.perspectiveTransform(self._cor, datapoint.get_homo())
            dist = 0.
            for p1,p2 in zip(des[0], desn[0]):
                dist += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            return dist
        else:
            return None

    def get_better_by_quality(self, datapoint):
        if self._quality > datapoint.get_quality():
            return self
        else:
            return datapoint

    def __str__(self):
        return "Method: " + str(self._method) + " Quality: " + str(self._quality) + " Marks:" + str(self._marks)

    def __mul__(self, other):
        #Dont fail gracefully
        if other.get_homo() is not None and self.get_homo() is not None:
            nhomo = np.dot(self.get_homo(), other.get_homo())
        else:
            nhomo = None

        return DataPoint(self._method + "*" + other._method, self._shape, self._quality * other._quality, nhomo )
class DataPoints(DataPoint):
    def __init__(self, points):
        #For compatibility
        DataPoint.__init__(self, points[0])
        self.points = points
    def __str__(self):
        return self.points
    def getIntrest(self):
        #We tolerate if differences are similar by factor or 200 (which is bigger)
        f = 1.2
        s = 0
        maxd = 0
        n = 0
        for p1 in self.points:
            for p2 in self.points:
                if p1 != p2:
                    dist = p1.get_distance(p2)
                    if dist is None:
                        return 1
                    s += dist
                    maxd = max(maxd, dist)
                    n +=1
        s /= 1.0*n
        toleration = min(f*s, 200)
        return 1./(maxd/toleration)


class FeatureExtractor(object):
    def __init__(self, method=None):
        self._detector = cv2.FeatureDetector_create(method)
        self._extractor = cv2.DescriptorExtractor_create(method)
        if method=="SURF":
            self._extractor.setInt("upright", 1)
            self._extractor.setInt("extended",0)
            self._extractor.setDouble("hessianThreshold",400)
            self._detector.setInt("upright", 1)
            self._detector.setInt("extended",0)
            self._detector.setDouble("hessianThreshold",400)
        if method=="SIFT":
            #self._detector.setDouble("contrastThreshold",0.08)
            #self._detector.setDouble("edgeThreshold",8)
            self._detector.setDouble("sigma",1.4)



    def getFeatures(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints = self._detector.detect(grey, None)
        ret = self._extractor.compute(grey, keypoints)
        return ret

class MatchDesciptorsFlann(object):
    FLANN_INDEX_LININD = 0
    FLANN_INDEX_KDTREE = 1

    def __init__(self, desc, method):
        method_to_option = {"SURF":1, "SIFT":1,"FAST":1,"ORB":0}
        self._option  = method_to_option[method]
        if self._option == self.FLANN_INDEX_LININD:
            nd = desc.view(np.float32).copy()
            desc = nd

        options = [{'algorithm': self.FLANN_INDEX_LININD},
                   {'algorithm': self.FLANN_INDEX_KDTREE, 'trees': 4}]
        self._flann = cv2.flann_Index(desc, options[method_to_option[method]])

    def getPairs(self, desc, r_threshold = 0.6):
        if self._option == self.FLANN_INDEX_LININD:
            nd = desc.view(np.float32).copy()
            desc = nd

        (idx2, dist) = self._flann.knnSearch(desc, 2, params={})
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc))
        pairs = np.int32(zip(idx1, idx2[:,0]))
        return pairs[mask]

