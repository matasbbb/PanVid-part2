import cv2
import numpy as np


class FeatureExtractor(object):
    def __init__(self, method=None):
        self._detector = cv2.FeatureDetector_create(method)
        self._extractor = cv2.DescriptorExtractor_create(method)

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
                rez.append((data[1][2],data[0][2]))
            else:
                rez.append(None)
        return rez


class RegisterImagesStandart2D(RegisiterImagesInterface):

    def getDiff(self, method="SURF"):
        lastframe = self._stream.getFrame()
        frame = self._stream.getFrame()
        extractor = FeatureExtractor(method)
        diff_2d = []
        while frame is not None:
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
                diff_2d.append(homography)
            else:
                diff_2d.append(None)
            lastframe = frame
            frame = self._stream.getFrame()
        return diff_2d



feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.0005,
                       minDistance = 6,
                       blockSize = 23 )

lk_params = dict( winSize  = (23, 23),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))


class RegisterImagesLK2D(RegisiterImagesInterface):
    _args = {"back_threshold":["Back threshold",float, 2.0]}

    def __init__(self, stream):
        self._stream = stream

    def getDiff(self, method="LK", back_threshold=2.0):
        if method is not "LK":
            print "Not implemented"
            return None
        frame_idx = 0
        diff_2d = []
        frame = self._stream.getFrame()
        prev_frame = self._stream.getFrame()
        p0 = None
        while frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            if p0 is None or frame_idx % 5 == 0 or len(p0) < 16:
                p0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            status = d < back_threshold
            p1 = p1[status].copy()
            p0 = p0[status].copy()
            #Not enougth points
            if len(p0) < 4:
                p0 = None
                diff_2d.append(None)
                prev_frame = frame
                frame = self._stream.getFrame()
                frame_idx += 1
                continue
            homography, status = cv2.findHomography(p1, p0, cv2.RANSAC)
            diff_2d.append(homography)
            prev_frame = frame
            frame = self._stream.getFrame()
            frame_idx += 1
        return diff_2d


