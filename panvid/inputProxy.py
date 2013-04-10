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

    def modifyDataPoints(self,datapoints):
        return datapoints

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

class StreamProxyResize(StreamProxy):
    #Prefered method is area, because most often we resample down, not zoom
    def __init__(self, stream, sizef=(0.5,0.5), method=cv2.cv.CV_INTER_AREA):
        self._sizef = sizef
        self._method = method
        self._fs = None
        self._stream = stream

    def getFrame(self):
        frame = self._stream.getFrame()
        if frame is None:
            return None
        nframe = cv2.resize(frame, None, self._sizef[0], self._sizef[1], self._method)
        if self._fs is None and frame is not None:
            self._fs = frame.shape
            self._nfs = nframe.shape
        return nframe

    def modifyDataPoints(self,datapoints):
        #TODO test if stacked, so first use prevous
        if self._fs is None:
            print "no framesize"
            return datapoints
        else:
            #FixThat
            return datapoints
            for d in datapoints:
                d[0,2] *= self._sizef[0]
                d[1,2] *= self._sizef[1]
            return datapoints
            nhomo = np.matrix([[0,0,0],[0,0,0],[0,0,0]])

    def getClone(self):
        return StreamProxyResize(self._stream.getClone(), self._sizef, self._method)

class StreamProxyCrop(StreamProxy):
    #Prefered method is area, because most often we resample down, not zoom
    def __init__(self, stream, size=(1000,1000), center=True):
        self._size = size
        self._center = center
        self._fs = None
        self._stream = stream

    def getFrame(self):
        frame = self._stream.getFrame()
        if frame is None:
            return None
        size = self._size
        if not self._center:
            nframe = frame[:size[0],:size[1],:]
        else:
            x = frame.shape[0]/2 - size[0]/2
            y = frame.shape[1]/2 - size[1]/2
            self._edges = (x,y,size[0]-x,size[1]-y)
            nframe = frame[x:x+size[0],y:y+size[1]]
        self._fs = frame.shape
        return nframe

    def modifyDataPoints(self,datapoints):
        #It requires editing if it is not corner
        if self._center:
            xt,yl,xb,yr = self._edges
            #Looks like weird switch over... WTF!
            yl,xt,yr,xb = self._edges
            #We got smaller ones homo, we need to convert to bigger!
            #So intrest points are around
            cor = np.array([[-xt,-yl],[-xt,yr],[xb,-yl],[xb,yr]], dtype='float32')
            cor = np.array([cor])
            for d in datapoints:
                if d._homo is not None:
                    des = cv2.perspectiveTransform(cor, d._homo)
                    homo, _ = cv2.findHomography(cor, des)
                    d._homo = homo
        return datapoints

    def getClone(self):
        return StreamProxyCrop(self._stream.getClone(), self._size, self._center)


