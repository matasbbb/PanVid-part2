import cv2
import numpy as np


class DataPoint():
    def __init__(self, method, quality=0, homo=None):
        self._method = method
        self._quality = quality
        self._homo = homo
        if homo is None:
            self._quality = 0

    def get_homo(self):
        return self._homo

    def get_naive2D(self):
        return (self._homo[1][2],self._homo[0][2])

    def get_quality(self):
        return self._quality

    def get_better_by_quality(self, datapoint):
        if self._quality > datapoint.get_quality():
            return self
        else:
            return datapoint

    def __str__(self):
        return "Method: " + str(self._method) + " Quality: " + str(self._quality)



class FeatureExtractor(object):
    def __init__(self, method=None):
        self._detector = cv2.FeatureDetector_create(method)
        self._extractor = cv2.DescriptorExtractor_create(method)
        if method=="SURF":
            self._extractor.setDouble("hessianThreshold",400)

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


registered_methods = {}

class RegisterImages():
    def __init__(self, stream):
        self._stream = stream

    def getDiff(self, *args):
        print "Not implemented getDiff!"
        return []


class RegisterImagesStandart(RegisterImages):
    def getDiff(self, method="SURF", fmask=None, doneCB=None, progressCB=None):
        lastframe = self._stream.getFrame()
        frame = self._stream.getFrame()
        extractor = FeatureExtractor(method)
        diff_2d = []
        frame_idx = 0
        while frame is not None:
            if progressCB is not None:
                overall, curr = self._stream.getProgress()
                if fmask is not None:
                    overall = sum(fmask)
                    curr = sum(fmask[:frame_idx])
                progressCB(overall, curr)

            if fmask is None or fmask[frame_idx]:
                #Find features
                (k1, d1) = extractor.getFeatures(lastframe)
                (k2, d2) = extractor.getFeatures(frame)
                #Match them
                flann = MatchDesciptorsFlann(d2, method)
                match = flann.getPairs(d1)
                c1 = np.array([k1[e].pt for e in match[...,0]], 'float32')
                c2 = np.array([k2[e].pt for e in match[...,1]], 'float32')
                #Find homography
                if (len(c2) >= 4):
                    homography, mask = cv2.findHomography(c2, c1, cv2.RANSAC)
                    #Asume if features matched it means they are good
                    quality = min(len(c2)/16,1)
                    #quality *= min(1,float(sum(mask))/len(mask) * 10)
                    diff_2d.append(DataPoint(method,quality,homography))
                else:
                    diff_2d.append(DataPoint(method))
            else:
                diff_2d.append(DataPoint(method))
            lastframe = frame
            frame = self._stream.getFrame()
            frame_idx += 1
        if doneCB is not None:
            doneCB(diff_2d)
        return diff_2d

registered_methods["SURF"] = RegisterImagesStandart
registered_methods["SIFT"] = RegisterImagesStandart
#registered_methods["FAST"] = RegisterImagesStandart
#registered_methods["ORB"] = RegisterImagesStandart

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.05,
                       minDistance = 40,
                       blockSize = 23,
                       useHarrisDetector = True)

lk_params = dict( winSize  = (23, 23),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005))

class RegisterImagesLK(RegisterImages):
    def __init__(self, *args):
        RegisterImages.__init__(self, *args)
        #self.detector = lambda image: cv2.FeatureDetector_create("FAST").detect(image)
        self.detector = lambda image: cv2.goodFeaturesToTrack(image, **feature_params)

    def getDiff(self, method="LK", fmask=None, doublecheck=False, doneCB=None, progressCB=None):
        if method != "LK":
            print "Not implemented in RegisterImagesLK " + method
            return None
        frame_idx = 0
        diff_2d = []
        prev_frame = self._stream.getFrame()
        frame = self._stream.getFrame()
        p0 = None
        while frame is not None:
            if progressCB is not None:
                overall, curr = self._stream.getProgress()
                if fmask is not None:
                    overall = sum(fmask)
                    curr = sum(fmask[:frame_idx])
                progressCB(overall, curr)

            if fmask is None or fmask[frame_idx]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                if p0 is None or frame_idx % 20 == 0 or len(p0) < 32:
                    p0 = self.detector(prev_gray)
                p1, st0, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, **lk_params)
                #If try back check, which reduces speed, but improves quality
                if doublecheck:
                    p0r, st1, err = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, **lk_params)
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
                    #Hack need real way!
                    back_threshold = 0.02
                    d = abs(p0).reshape(-1, 2).max(-1)
                    status = True
                    status = np.logical_and(st0.flatten(),status)
                p1 = p1[status].copy()
                p0 = p0[status].copy()


                #If not enougth points
                if len(p0) < 4:
                    p0 = None
                    diff_2d.append(DataPoint(method))
                else:
                    #print str(quality) + " "+str(sum(st)) + " " + str(len(p0)) +  " " + str(back_threshold)
                    homography, mask = cv2.findHomography(p1, p0, cv2.RANSAC)
                    quality = min(1, 0.02/back_threshold) * min(len(p0)/16, 1)
                    quality *= mask.sum()/mask.size

                    #quality *= min(1,(float(sum(mask))/len(mask)*10)
                    diff_2d.append(DataPoint(method,quality,homography))
                    #Predicted points in new frame moved
                    p0 = p1
            else:
                p0 = None
                diff_2d.append(DataPoint(method))

            prev_frame = frame
            frame = self._stream.getFrame()
            frame_idx += 1

        if doneCB is not None:
            doneCB(diff_2d)
        return diff_2d
registered_methods["LK"] = RegisterImagesLK

class RegisterImagesDetect(RegisterImages):
    def getDiff(self, method="LK-SIFT", fmask=None, quality = 0.8, comp=True, doneCB=None, progressCB=None):
        methods= method.split("-")
        fmask = None
        old_rez = None
        for m in methods:
            progressCB_pass = None
            if progressCB is not None:
                progressCB_pass = lambda overall, curr: progressCB (overall, \
                        curr, len(methods), methods.index(m) + 1)
            if old_rez is not None:
                fmask = []
                for d in rez:
                    fmask.append(d.get_quality() < quality)
            if not registered_methods.has_key(m):
                print "Not implemented " + m
                return None
            reg = registered_methods[m](self._stream.getClone())
            rez = reg.getDiff(m, fmask, progressCB=progressCB_pass)
            #If we have old rezults merge
            if old_rez is not None:
                new_rez = []
                for old_d, d in zip(old_rez,rez):
                    # Get recent if we dont want to compare ant it is not None
                    # And if we want to compare and both are not None
                    if comp:
                        new_rez.append(d.get_better_by_quality(old_d))
                    else:
                        if d.get_homo() is not None:
                            new_rez.append(d)
                        else:
                            new_rez.append(old_d)
                rez = new_rez

            old_rez = rez
        if doneCB is not None:
            doneCB(old_rez)
        return old_rez
