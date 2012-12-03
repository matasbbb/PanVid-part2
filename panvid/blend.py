import cv2
import numpy as np


class BlendInterface():
    def __init__(self, stream):
        self._stream = stream
        #First frame as base
        self._pano = stream.getFrame()
        self._window = []
        self._data = None

    def getPano(self):
        return self._pano

    def prevPano(self, wait=True, window="Preview", image=None):
        if image is None:
            image = self._pano
        if window not in self._window:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if len(self._window) == 0:
                cv2.startWindowThread()
            self._window.append(window)
        cv2.imshow(window, image)
        while cv2.waitKey(100) != 27 and wait:
            cv2.imshow(window, image)

    def blendNextN (self, dataset, prev=False):
        for data in dataset:
            self.blendNext(data)
            if prev:
               self.prevPano()

class BlendOverlay2D(BlendInterface):
    def blendNext(self, shift):
        if shift is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            shift = (int(round(shift[0])), int(round(shift[1])))
            print shift
            image = self._stream.getFrame()
            h,w,_ = image.shape
            hp,wp,_ = self._pano.shape
            if self._data is None:
                self._data = shift
            else:
                self._data = (self._data[0] + shift[0],
                              self._data[1] + shift[1])
            shift = self._data
            print shift
            add = [0,0,0,0]
            #If negative move
            add[2] = max(-shift[1], 0)
            add[0] = max(-shift[0], 0)
            shift = (max(0,shift[0]),max(0,shift[1]))
            #If too small, make bigger
            add[3] = max(0, shift[1] + w - wp)
            add[1] = max(0, shift[0] + h - hp)
            print add
            #First and add boundaries that new image will fit
            self._pano = cv2.copyMakeBorder(self._pano, *add, borderType=cv2.BORDER_CONSTANT)
            print shift
            self._pano[shift[0]:shift[0]+h, shift[1]:shift[1]+w] = image
            #If moved shift changed
            self._data = shift

class BlendOverlayHomo(BlendInterface):
    def blendNext(self, homo):
        if homo is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            if self._data is None:
                self._data = homo
            else:
                self._data = np.dot(self._data, homo)
            homography = self._data
            image2 = self._stream.getFrame()
            image1 = self._pano
            ## Work out position of image corners after alignment
            (h1,w1,_) = image1.shape
            (h2,w2,_) = image2.shape
            p1 = np.array([[0, 0, w1-1, w1-1], [0, h1-1, 0, h1-1]])
            p2 = np.array([[0, 0, w2-1, w2-1], [0, h2-1, 0, h2-1], [1, 1, 1, 1]])
            p2 = np.matrix(homography) * p2;
            p2 = p2[0:2,...] / p2[2,...].repeat(2, axis=0)
            p = np.concatenate((p1, p2), axis=1)

            ## Calculate translation and size of output bitmap
            print p
            t = -p.min(1)
            print t
            s = np.ceil(p.max(1) + t).astype(np.int32)
            t = (t[0,0], t[1,0])
            s = (s[0,0], s[1,0])
            print "output size: %ix%i" % s

            ## Translate everything
            homography[0:2,2] += t

            ## Warp second image to fit the first
            pano = cv2.warpPerspective(image2, homography, s)
            self.prevPano(window="Debug", image=pano)
            pano[t[1]:h1+t[1], t[0]:w1+t[0]] = image1
            self._pano = pano
            self._data = homography
