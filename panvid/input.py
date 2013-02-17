import cv2
import numpy as np

class StreamInput(object):
    def __init__(self):
        self._framenum = 0

    def skipFrames(self, n):
        self._framenum += n

    def getProgress(self):
        return 0

InputRegister = {}


class VideoInput(StreamInput):
    def __init__(self, url):
        StreamInput.__init__(self)
        self._url = url
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

    def getProgress(self):
        frame_count = self._capt.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        curr_frame = self._capt.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        return (frame_count, curr_frame)


    def getAlphaFrame(self):
        f = self.getFrame()
        if f is not None:
             f = cv2.cvtColor(f, cv2.cv.CV_BGR2BGRA)
        return f

    def getClone(self):
        return VideoInput(self._url)


InputRegister["Simple video"] = (VideoInput, "video.glade")

class VideoInputAdvanced(VideoInput):
    def __init__(self, url, bound=0, skip=0, start=0):
        VideoInput.__init__(self, url)
        self._bound = int(bound)
        self._skip = int(skip)
        self.skipFrames(int(start))
        self._start = int(start)

    def getFrame(self):
        if self._framenum > self._bound and self._bound is not 0:
            return None
        (succ, frame) = self._capt.read()
        if succ:
            self._framenum += 1
            self.skipFrames(self._skip)
            return frame
        else:
            return None

    def getClone(self):
        return VideoInputAdvanced(self._url, self._bound, self._skip, self._start)

    def getProgress(self):
        if self._bound is 0:
            return super(VideoInputAdvanced, self).getProgress()
        else:
            return (self._bound, self._framenum)

InputRegister["Video with options"] = (VideoInputAdvanced, "videoskip.glade")

class ImageInput(StreamInput):
    def __init__(self, img_list):
        StreamInput.__init__(self)
        self._img_list = img_list
        self._org_img_list = img_list

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

    def getClone(self):
        return ImageInput(self._org_img_list)

    def getProgresS(self):
        return (len(self._org_img_list),len(self._img_list))
#InputRegister["Image list"] = (ImageInput, "image.glade")
