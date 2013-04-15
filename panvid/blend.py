import logging
import numpy as np
import PIL.Image as PImg
import math
from panvid.predictHelpers import *
from scipy import  weave
BlendRegister = {}
class BlendImages():
    def __init__(self, stream=None, reg=None, progressCB=None, **args):
        self._args = args
        self._log = logging.getLogger(__name__)
        self._log.info("Blender created")
        self._stream = stream
        self._reg = reg
        if reg is not None:
            self._stream = MockStream()
            self._stream.setFrame(reg._frames[0])
        #First frame as base
        self._pano = self._stream.getFrame()
        self._window = []
        self._data = None
        self._pg=progressCB
        #self.prevPano()
        self.methodInit(**args)

    def methodInit(self, **args):
        return

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
        if self._reg is not None:
            d = self._reg.getNextDataPoint()
            while d is not None:
                self._stream.setFrame(self._reg._frames[0])
                rez = self.blendNext(d)
                if rez is not None:
                    return
                if self._pg is not None:
                    self._pg(self._reg.getProgress())
                if prev:
                    self.prevPano(wait=wait)
                d = self._reg.getNextDataPoint()
        else:
            for data in dataset:
                self._pg(self._stream.getProgress())
                rez = self.blendNext(data)
                if rez is not None:
                    if prev:
                       self.prevPano(wait=wait)


class BlendOverlayCInline(BlendImages):
    def methodInit(self, warp=False, **args):
        if self._mtype is not None:
            self._mask = self.createMask(self._mtype)
            if self._weights is not None:
                self._weights = self._mask.copy()
        if self._imtype is not None:
            self._pano = self._pano.astype(imtype)  
        self._2D = not warp

    def createMask(self, mtype):
        size = np.array(self._pano.shape)[:2]
        mask = np.zeros(size, mtype)
        midx, midy = int(size[0]/2), int(size[1]/2)
        maxx, maxy = int(size[0]), int(size[1])
        code = """
            for (int x = 0; x < maxx; x++)
                for (int y = 0; y < maxy; y++)""" + self.maskCode
        weave.inline(code,['maxx','maxy','midx','midy','mask'],
                    type_converters=weave.converters.blitz, compiler="gcc")
        return mask

    def getFrame(self, dataPoint):
        image = self._stream.getFrame()
        if self._2D:  
            shift = dataPoint.get_better2D()
            shift = [int(round(shift[0])), int(round(shift[1]))]
            if self._data is None:
                self._data = shift
            else:
                self._data = (self._data[0] + shift[0],
                              self._data[1] + shift[1])
            shift = self._data
            return image, shift

    def addBorders(self, shift, fshape, over=40):
        add = [0,0,0,0]
        h,w,_ = fshape
        hp,wp,_ = self._pano.shape

        #If negative move
        add[2] = max(-shift[1], 0)*over
        add[0] = max(-shift[0], 0)*over
        #If too small, make bigger
        add[3] = max(0, shift[1] + w - wp)*over
        add[1] = max(0, shift[0] + h - hp)*over
        nshift = (shift[0]+add[0], shift[1]+add[2])
        if add != [0,0,0,0]:
            print "copy " + str(add)
            self._pano = cv2.copyMakeBorder(self._pano, *add, borderType=cv2.BORDER_CONSTANT)
            if self._weights is not None: 
                nweights = np.zeros(self._pano.shape[:2], dtype=self._mtype)
                if self._weights.min() != 0:
                    nweights += self._weights.min()-1
                nweights[add[0]:self._weights.shape[0]+add[0],
                         add[2]:self._weights.shape[1]+add[2]]=self._weights
                self._weights = nweights
        self._data = nshift
        return nshift

    def getMask(self, dataPoint):
        if self._2D:
            return self._mask

    def add(self, shift, image, mask):
            h,w,_ = image.shape
            x1,x2,y1,y2 = shift[0],shift[0]+h, shift[1],shift[1]+w
            #Cant access object variables
            mask, weights, pano = self._mask, self._weights, self._pano
            code ="""
                for(int x = x1; x < x2; x++)
                    for(int y = y1; y < y2; y++)
                               """ + self.blendCode
            weave.inline(code,['x1','x2','y1','y2',
                               'mask','weights','pano','image'],
                         type_converters=weave.converters.blitz, 
                         compiler="gcc")

    def blendNext(self, dataPoint):
        if dataPoint is None or dataPoint._homo is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            frame, shift = self.getFrame(dataPoint)
            mask = self.getMask(dataPoint)
            shift = self.addBorders(shift, frame.shape)
            self.add(shift, frame, mask) 




class BlendOverlayNearest(BlendOverlayCInline):
    _weights = True
    _mtype = np.int
    _imtype = None
    maskCode = "mask(x,y) = 4294967296 - (x-midx)*(x-midx) - (y-midy)*(y-midy);"
    blendCode =  """
                        if ( weights(x,y) < mask(x-x1,y-y1)){
                            weights(x,y) = mask(x-x1,y-y1);
                            pano(x,y,0) = image(x-x1,y-y1,0);
                            pano(x,y,1) = image(x-x1,y-y1,1);
                            pano(x,y,2) = image(x-x1,y-y1,2);
                            }
                        """

    
BlendRegister["Overlay nearest"] = (BlendOverlayNearest, None)




class BlendOverlay2D(BlendImages):
    def blendNext(self, dataPoint):
        if dataPoint is None or dataPoint._homo is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            shift = dataPoint.get_better2D()
            if self._args.has_key("limit") and False:
                shift = [shift[0]*self._args["limit"][0], shift[1]*self._args["limit"][1]]
            shift = [int(round(shift[0])), int(round(shift[1]))]
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
            #print add
            #First and add boundaries that new image will fit
            self._pano = cv2.copyMakeBorder(self._pano, *add, borderType=cv2.BORDER_CONSTANT)
            self._pano[shift[0]:shift[0]+h, shift[1]:shift[1]+w] = image
            #If moved shift changed
            self._data = shift

class BlendOverlayWeight(BlendImages):
    _mask = None
    def getPano(self):
        #we need to normalise
        retpano = self._pano.copy()
        retpano[:,:,0] /= self._weights
        retpano[:,:,1] /= self._weights
        retpano[:,:,2] /= self._weights
        print retpano.max()
        return retpano.astype(np.uint8)

    def blendNext(self, dataPoint):
        tp = np.int
        if self._mask is None:
            print "creating mask"
            s and self.cropi
                     
        """ toImg size 
         def getMask(self):
             should be equal to fromImg"""
        if over:
            x,y,_ = self._pano.shape
            div = size[0]+size[1]            
            mask = np.zeros((size[0],size[1]),tp)
            for x in xrange(1,size[0]-1):
                for y in xrange(1,size[1]-1):
                        mask[x][y] = 1 + min(mask[x-1][y], mask[x][y-1])
            for x in xrange(size[0]-2,0,-1):
                for y in xrange(size[1]-2,0,-1):
                        mask[x][y] = min(mask[x][y],1 + min(mask[x+1][y], mask[x][y+1]))
            #normalise to 255
            #better?
            omask = mask.copy()
            for i in xrange(10):
                mask *= omask
                if mask.max() > 2**16:
                    mask /= 255

            #mask = mask*(2**16-2)/mask.max()
            big_values = mask > (2**16-2)
            mask[big_values] = (2**16-2)
            #   all image anyway!
            self._mask = (mask*(2**16)/mask.max())
            self._mask = mask + 1
            print "done mask max:" + str(mask.max())
            #self.prevPano(image=self._mask.astype(np.uint16))
            #self.prevPano(image=self._mask.astype(np.uint8))
    
            self._weights = self._mask.copy()
            self._pano = self._pano.astype(tp)
                  
            if dataPoint is None or dataPoint._homo is None:
                print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            shift = dataPoint.get_better2D()
            shift = [int(round(shift[0])), int(round(shift[1]))]
            print self._pano.shape
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
            over = 40
            #If negative move
            add[2] = max(-shift[1], 0)*over
            add[0] = max(-shift[0], 0)*over
            #If too small, make bigger
            add[3] = max(0, shift[1] + w - wp)*over
            add[1] = max(0, shift[0] + h - hp)*over
            shift = (shift[0]+add[0], shift[1]+add[2])
            #print add
            #First and add boundaries that new image will fit
            if add != [0,0,0,0]:
                print "copy " + str(add)
                self._pano = cv2.copyMakeBorder(self._pano, *add, borderType=cv2.BORDER_CONSTANT)
                self._weights = cv2.copyMakeBorder(self._weights, *add, borderType=cv2.BORDER_CONSTANT)

           
            #Just add image and weights
            #each bit is 2**32, we add max of 2**8*2**8, so we can add 2**16 frames...
            image = image.astype(tp)
            image[:,:,0] *= self._mask
            image[:,:,1] *= self._mask
            image[:,:,2] *= self._mask

            self._pano[shift[0]:shift[0]+h, shift[1]:shift[1]+w] += image
            self._weights[shift[0]:shift[0]+h, shift[1]:shift[1]+w] += self._mask
            #If moved shift changed
            self._data = shift

BlendRegister["Overlay by shift"] = (BlendOverlayWeight, None)
class BlendOverlay2D(BlendImages):
    def blendNext(self, dataPoint):
        if dataPoint is None or dataPoint._homo is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            shift = dataPoint.get_better2D()
            if self._args.has_key("limit") and False:
                shift = [shift[0]*self._args["limit"][0], shift[1]*self._args["limit"][1]]
            shift = [int(round(shift[0])), int(round(shift[1]))]
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
            #print add
            #First and add boundaries that new image will fit
            self._pano = cv2.copyMakeBorder(self._pano, *add, borderType=cv2.BORDER_CONSTANT)
            self._pano[shift[0]:shift[0]+h, shift[1]:shift[1]+w] = image
            #If moved shift changed
            self._data = shift

BlendRegister["Overlay by shift"] = (BlendOverlay2D, None)
class BlendOverlayHomo(BlendImages):
    def __init__(self, *args):
        BlendImages.__init__(self, *args)

    def blendNext(self, datapoint):
        homo = datapoint.get_homo()
        if homo is None:
            print "Image skiped, expect artifacts"
            nframe = self._stream.getFrame()
        else:
            if self._data is None:
                self._data = homo
            else:
                self._data = np.dot(self._data, homo)
            homography = self._data
            nframe = self._stream.getFrame()
            pano = self._pano

            ## Work out position of image corners after alignment
            (ph,pw,_) = pano.shape
            (fh,fw,_) = nframe.shape
            np.set_printoptions(precision=3, suppress=True)
            cor = np.array([[0,0],[0, fh-1],[fw-1,0],[fw-1,fh-1]],dtype='float32')
            cor = np.array([cor])
            corhom = cv2.perspectiveTransform(cor, homography)[0]
            
            #Crop square
            crop = 3
            #Want to make trans image such size that all filled after transform
            top = int(max(corhom[0][1],corhom[2][1]))+crop
            bot = int(min(corhom[1][1],corhom[3][1]))-crop
            left = int(max(corhom[0][0],corhom[1][0]))+crop
            right = int(min(corhom[2][0], corhom[3][0]))-crop
              
            #Now we get image which we need to put at (top, left, right, bot)
            add=[0,0,0,0]
            if top < 0:
                add[0] = -top
                homography[1][2] += -top
                bot += -top
                top = 0
            if left < 0:
                add[2] = -left
                homography[0][2] += -left
                right += -left
                left = 0
            if right > pw:
                add[3] = right - pw
            if bot > ph:
                add[1] = bot - ph
            
            #We want to transform to 0,0
            chomo = np.dot(np.matrix([[1,0,-left],[0,1,-top],[0,0,1]]), homography)
            s = (right-left, bot-top)
            if s[0] < 0 or s[1] < 0:
                return False

            transf_image = cv2.warpPerspective(src=nframe,M=chomo, dsize=s)
            pano = cv2.copyMakeBorder(pano, *add, borderType=cv2.BORDER_CONSTANT)
            pano[top:bot,left+cropl:right+cropr] = transf_image[:,cropl:-cropr]

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

