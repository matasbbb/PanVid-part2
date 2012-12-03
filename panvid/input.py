import cv2
import numpy as np


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

class VideoInputSkip(VideoInput):
    def __init__(self, url, bound, skip):
        VideoInput.__init__(self, url)
        self._bound = bound
        self._skip = skip
    def getFrame(self):
        if self._framenum > self._bound:
            return None
        (succ, frame) = self._capt.read()
        if succ:
            self._framenum += 1
            self.skipFrames(self._skip)
            print self._framenum
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

