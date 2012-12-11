import cv2
import numpy as np


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
    def __init__(self, desc):
        FLANN_INDEX_KDTREE = 1
        self._flann = cv2.flann_Index(desc,
                {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

    def getPairs(self, desc, r_threshold = 0.6):
        (idx2, dist) = self._flann.knnSearch(desc, 2, params={})
        mask = dist[:,0] / dist[:,1] < r_threshold
        idx1 = np.arange(len(desc))
        pairs = np.int32(zip(idx1, idx2[:,0]))
        return pairs[mask]


registered_methods = {}

class RegisiterImagesInterface():
    def __init__(self, stream):
        self._stream = stream

    def getDiff(self, *args):
        print "Not implemented getDiff!"
        return []

    def getDiff2D(self, *args):
        homo = self.getDiff(*args);
        rez = []
        for data in homo:
            if data is not None:
                q, data = data
                rez.append((q,(data[1][2],data[0][2])))
            else:
                rez.append(None)
        return rez


class RegisterImagesStandart2D(RegisiterImagesInterface):
    def getDiff(self, method="SURF", fmask=None):
        lastframe = self._stream.getFrame()
        frame = self._stream.getFrame()
        extractor = FeatureExtractor(method)
        diff_2d = []
        frame_idx = 0
        while frame is not None:
            if fmask is None or fmask[frame_idx]:
                #Find features
                (k1, d1) = extractor.getFeatures(lastframe)
                (k2, d2) = extractor.getFeatures(frame)
                #Match them
                flann = MatchDesciptorsFlann(d2)
                match = flann.getPairs(d1)
                c1 = np.array([k1[e].pt for e in match[...,0]], 'float32')
                c2 = np.array([k2[e].pt for e in match[...,1]], 'float32')
                #Find homography
                if (len(c2) >= 4):
                    homography, mask = cv2.findHomography(c2, c1, cv2.RANSAC)
                    #Asume if features matched it means they are good
                    quality = min(len(c2)/16,1)
                    #quality *= min(1,float(sum(mask))/len(mask) * 10)
                    diff_2d.append((quality,homography))
                else:
                    diff_2d.append(None)
            else:
                diff_2d.append(None)
            lastframe = frame
            frame = self._stream.getFrame()
            frame_idx += 1
        return diff_2d

registered_methods["SURF"] = RegisterImagesStandart2D
registered_methods["SIFT"] = RegisterImagesStandart2D

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.005,
                       minDistance = 6,
                       blockSize = 23 )

lk_params = dict( winSize  = (23, 23),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005))

class RegisterImagesLK2D(RegisiterImagesInterface):
    def getDiff(self, method="LK", fmask=None):
        if method != "LK":
            print "Not implemented in RegisterImagesLK2D " + method
            return None
        frame_idx = 0
        diff_2d = []
        prev_frame = self._stream.getFrame()
        frame = self._stream.getFrame()
        p0 = None
        while frame is not None:
            if fmask is None or fmask[frame_idx]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                if p0 is None or frame_idx % 5 == 0 or len(p0) < 16:
                    p0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

                p1, st0, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, **lk_params)
                p0r, st1, err = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, **lk_params)
                st = np.logical_and(st0.flatten(),st1.flatten())
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                back_threshold = 10.00
                #While we have enouth points reduce threshold
                while sum(np.logical_and(d < (back_threshold/2), st)) >= 16 and back_threshold/2 > 0.01:
                    back_threshold /= 2
                status = d < back_threshold
                status = np.logical_and(st,status)
                p1 = p1[status].copy()
                p0 = p0[status].copy()
                #Not enougth points
                if len(p0) < 4:
                    p0 = None
                    diff_2d.append(None)
                else:
                    #Biger threshold better, but we still want more than 8 points
                    #print str(quality) + " "+str(sum(st)) + " " + str(len(p0)) +  " " + str(back_threshold)
                    homography, mask = cv2.findHomography(p1, p0, cv2.RANSAC)
                    quality = min(1, 0.02/back_threshold) * min(len(p0)/16, 1)
                    #quality *= min(1,(float(sum(mask))/len(mask)*10)
                    diff_2d.append((quality,homography))
                    #Predicted points in new frame moved
                    p0 = p1
            else:
                diff_2d.append(None)

            prev_frame = frame
            frame = self._stream.getFrame()
            frame_idx += 1
        return diff_2d
registered_methods["LK"] = RegisterImagesLK2D

class RegisterImagesDetect(RegisiterImagesInterface):
    def getDiff(self, method="LK-SIFT", fmask=None, quality = 0.6, comp=True):
        methods= method.split("-")
        fmask = None
        old_rez = None
        for m in methods:
            if old_rez is not None:
                fmask = []
                for d in rez:
                    if d is not None:
                        q,r = d
                        fmask.append(q < quality)
                    else:
                        #If not found try with new method
                        fmask.append(True)

            if not registered_methods.has_key(m):
                print "Not implemented " + m
                return None
            reg = registered_methods[m](self._stream.getClone())
            rez = reg.getDiff(m, fmask)
            #If we have old rezults merge
            if old_rez is not None:
                new_rez = []
                for old_d, d in zip(old_rez,rez):
                    # Get recent if we dont want to compare ant it is not None
                    # And if we want to compare and both are not None
                    if (not comp and d is not None) \
                       or (comp and d is not None and old_d is not None and d[0] > old_d[0])\
                       or (comp and old_d is None):
                        new_rez.append(d)
                    else:
                        new_rez.append(old_d)
                rez = new_rez

            old_rez = rez
        return old_rez
