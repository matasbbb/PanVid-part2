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

class StreamInput(object):
    def __init__(self):
        self._framenum = 0

    def skipFrames(self, n):
        self._framenum += n

class VideoInput(StreamInput):
    def __init__(self, url):
        StreamInput.__init__(self)
        self._capt = cv2.VideoCapture(url)

    def skipFrames(self, n):
        for i in xrange(n):
            self._capt.grab()
        self._framenum += n

    def getFrame(self):
        (succ, frame) = self._capt.read()
        if succ:
            self._framenum += 1
            return frame
        else:
            return None

class ImageInput(StreamInput):
    def __init__(self, img_list):
        StreamInput.__init__(self)
        self._img_list = img_list

    def skipFrames(self, n):
        self._img_list[n:]
        self.framenum += n

    def getFrame(self):
        if len(self._img_list) == 0:
            return None
        frame = cv2.imread(self._img_list[0])
        self._img_list[0] = self._img_list[1:]
        self.framenum += 1
        return frame

class RegisterImagesStandart2D():
    def __init__(self, stream):
        self._stream = stream

    def getDiff2D(self, method="SURF"):
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
            homography, mask = cv2.findHomography(c2, c1, cv2.RANSAC)
            xdiff = homography[1][2]
            ydiff = homography[0][2]
            diff_2d.append((xdiff, ydiff))
            lastframe = frame
            frame = self._stream.getFrame()
        return diff_2d





