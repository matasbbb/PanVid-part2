import cv2
import logging
import numpy as np
import PIL.Image as PImg
import math

BlendRegister = {}
class BlendImages():
    def __init__(self, stream):
        self._log = logging.getLogger(__name__)
        self._log.info("Blender created")
        self._stream = stream
        #First frame as base
        self._pano = stream.getFrame()
        self._window = []
        self._data = None
        #self.prevPano()

    def setParams(self):
        return

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

    def blendNextN (self, dataset, prev=False, wait=False):
        for data in dataset:
            self.blendNext(data)
            if prev:
               self.prevPano(wait=wait)

    def merge(self, toImg, fromImg, over=True):
        """ toImg size should be equal to fromImg"""
        if over:
            toImg = fromImg
        else:
            for i in xrange(fromImg.size):
                toImg.flat[i] = max(toImg.flat[i],fromImg.flat[i])


class BlendOverlay2D(BlendImages):
    def blendNext(self, dataPoint):
        if dataPoint is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            shift = dataPoint.get_naive2D()
            shift = (int(round(shift[0])), int(round(shift[1])))
            image = self._stream.getFrame()
            h,w,_ = image.shape
            hp,wp,_ = self._pano.shape
            if self._data is None:
                self._data = shift
            else:
                self._data = (self._data[0] + shift[0],
                              self._data[1] + shift[1])
            shift = self._data
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

            #self.merge(self._pano[shift[0]:shift[0]+h, shift[1]:shift[1]+w],image,over=True)
            self._pano[shift[0]:shift[0]+h, shift[1]:shift[1]+w] = image
            #If moved shift changed
            self._data = shift

BlendRegister["Overlay by shift"] = (BlendOverlay2D, None)
class BlendOverlayHomo(BlendImages):
    def __init__(self, stream):
        BlendImages.__init__(self, stream)
        #self._pano = cv2.cvtColor(self._pano, cv2.cv.CV_BGR2BGRA)

    def blendNext(self, datapoint):
        homo = datapoint.get_homo()
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
            t = -p.min(1)
            s = np.ceil(p.max(1) + t).astype(np.int32)
            t = (t[0,0], t[1,0])
            # hack with max, due to 1px miscal
            s = (max(s[0,0], int(w1+t[0])), max(s[1,0], int(h1+t[1])))
            print "output size: %ix%i" % s

            ## Translate everything back
            homography[0:2,2] += t

            ## Warp second image to fit the first
            pano = cv2.warpPerspective(image2, homography, s, flags=cv2.INTER_CUBIC)
            self.prevPano(window="Debug", image=pano)
            #self.merge(pano[t[1]:h1+t[1], t[0]:w1+t[0]], image1, True)
            pano[t[1]:h1+t[1], t[0]:w1+t[0]] = image1


            self._pano = pano
            self._data = homography

BlendRegister["Overlay by homographies"] = (BlendOverlayHomo, None)

class BlendOverlayHomoWeight(BlendImages):
    #Functions which map from two arguments [0,1] to [0,1]
    posible_f={
        "radial":lambda x,y: math.sqrt(x**2+y**2),
        "h_inv_sqr":lambda x,y: 1-(1-abs(y))**2,
        "nasty_step":lambda x,y: 1-(1.1-math.sqrt((x/4)**2+y**2))**20,
        "h_step":lambda x,y: y>0.2 and y < 0.8
    }

    def __init__(self, stream):
        BlendImages.__init__(self, stream)
        self._pano = cv2.cvtColor(self._pano, cv2.cv.CV_BGR2BGRA)
        self._masks = {}
        self.l = self.posible_f["h_step"]
        self.blur = (7,7)

    def setParams(self, l=posible_f["h_step"], blur=None):
        if self.posible_f.has_key(l):
            l = posible_f(l)
        self.l = l
        self.blur = b

    def getAlphaMask(self, size):
        if self._masks.has_key(size):
            return self._masks[size]
        else:
            mask = np.zeros((size[0],size[1]),np.uint8)
            div = max(size[0]/2,size[1]/2)
            for x in xrange(size[0]):
                for y in xrange(size[1]):
                    r = self.l(1.0*x/size[0]-0.5,1.0*y/size[1]-0.5)
                    mask[x,y] = min(max(0, 255 - int(255*r)),255)
            if self.blur is not None:
                mask = cv2.GaussianBlur(mask, self.blur, -1,
                        borderType=cv2.BORDER_CONSTANT)
            self._masks[size] = mask
            self.prevPano(wait=False, window="alpha", image=mask)
            return mask

    def blendNext(self, datapoint):
        homo = datapoint.get_homo()
        if homo is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            if self._data is None:
                self._data = homo
            else:
                self._data = np.dot(self._data, homo)
            homography = self._data
            image2 = self._stream.getAlphaFrame()
            image2[:,:,3] = self.getAlphaMask((image2.shape[0], image2.shape[1]))
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
            t = -p.min(1)
            s = np.ceil(p.max(1) + t).astype(np.int32)
            t = (t[0,0], t[1,0])
            # hack with max, due to 1px miscal
            s = (max(s[0,0], int(w1+t[0])), max(s[1,0], int(h1+t[1])))
            print "output size: %ix%i" % s

            ## Translate everything back
            homography[0:2,2] += t

            ## Warp second image to fit the first
            pano = cv2.warpPerspective(image2, homography, s, flags=cv2.INTER_LANCZOS4)
            #self.prevPano(window="Debug", image=pano[:,:,3])
            #self.merge(pano[t[1]:h1+t[1], t[0]:w1+t[0]], image1, True)

            #Now we create same size images
            image1b = np.zeros((pano.shape[0], pano.shape[1],4),np.uint8)
            image1b[t[1]:h1+t[1], t[0]:w1+t[0]] = image1
            #self.prevPano(window="Alpha old", image=image1b[:,:,3])
            #self.prevPano(window="Alpha over", image=pano[:,:,3])
            #new alpha just map where we have pixels
            newalpha = (pano[:,:,3]/2) + (image1b[:,:,3] != 0) * 128
            #If in old panorama there are now old pixels just use from new
            blendalpha = (pano[:,:,3] * (image1b[:,:,3] != 0)) + (image1b[:,:,3] == 0)*255
            #blendalpha = cv2.blur(blendalpha, (20,20))
            pano[:,:,3] = blendalpha
            #self.prevPano(window="Alpha over", image=pano[:,:,3])
            #Merge using pil
            img = PImg.fromarray(image1b)
            pano1 = PImg.fromarray(pano)
            pano2 = PImg.composite(pano1, img, pano1)
            panon = np.array(pano2)
            panon[:,:,3] = newalpha
            #self.prevPano(window="pano alpha", image=panon[:,:,3])

            self._pano = panon
            self._data = homography

BlendRegister["Weighted overlay"] = (BlendOverlayHomoWeight, "options.glade")

