import cv2
import numpy as np
from panvid.input import StreamInput
class StreamProxy(StreamInput):
    def __init__(self, stream):
        self._stream = stream

    def getClone(self):
        return StreamProxy(self._stream.getClone())

    def getProgress(self):
        return self._stream.getProgress()

class StreamProxyBorder(StreamProxy):
    def __init__(self, stream, borderColor=(0,0,255), borderWidth=5,
            borderColorEnd=(0,0,50)):
        self._bc = borderColor
        self._bw = borderWidth
        self._bce = borderColorEnd
        self._stream = stream

    def getFrame(self):
        frame = self._stream.getFrame()
        for z in xrange(self._bw, 0, -1):
            col = np.array(self._bc) - \
                  (np.array(self._bc) - np.array(self._bce)) \
                  /(self._bw)*(self._bw-z)
            frame[0:z,:] = col
            frame[:,0:z] = col
            x = frame.shape[0]
            y = frame.shape[1]
            frame[x-z:x,:] = col
            frame[:,y-z:y] = col
        return frame

    def getClone(self):
        return StreamProxyBorder(self._stream.getClone(), self._bc, self._bw)
