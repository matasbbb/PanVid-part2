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
        self.stop = False
        if self._reg is not None:
            d = self._reg.getNextDataPoint()
            while d is not None and not self.stop:
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
                if self.stop:
                    return
                if self._pg is not None:
                    self._pg(self._stream.getProgress())
                rez = self.blendNext(data)
                if rez is not None:
                    if prev:
                       self.prevPano(wait=wait)


class BlendOverlayCInline(BlendImages):
    def methodInit(self, warp=False, **args):
        self._2D = not warp
        if not hasattr(self, "_support"):
            self._support = None
        if self._mtype is not None:
            self._mask = self.createMask()
            if self._weights is not None:
                self._weights = self._mask.copy()
        if self._imtype is not None:
            self._pano = self._pano.astype(self._imtype)  

    def createMask(self):
        size = np.array(self._pano.shape)[:2]
        mask = np.zeros(size, self._mtype)
        midx, midy = int(size[0]/2), int(size[1]/2)
        maxx, maxy = int(size[0]), int(size[1])
        code = """
            for (int x = 0; x < maxx; x++)
                for (int y = 0; y < maxy; y++)""" + self.maskCode
        weave.inline(code,['maxx','maxy','midx','midy','mask'],
                    type_converters=weave.converters.blitz, 
                    support_code=self._support,
                    compiler="gcc")
        if hasattr(self, "maxmask") and mask.max() > self.maxmask:
            mask = mask*self.maxmask/mask.max()
        if not self._2D and self._mtype == np.int32:
            self._mtype = np.double
            mask = mask.astype(np.double)
        return mask

    def getPano(self):
        return self._pano

    def getFrameMask(self, dataPoint):
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
            return image, shift, self._mask
        else:
            ## Work out position of image corners after alignment
            if self._data  is None:
                self._homo = dataPoint._homo
                self._data = [0,0]
            else:
                self._homo = np.dot(dataPoint._homo, self._homo)
            (ph,pw,_) = self._pano.shape
            (fh,fw,_) = image.shape
            cor = np.array([[0,0],[0, fh-1],[fw-1,0],[fw-1,fh-1]],dtype='float32')
            cor = np.array([cor])
            corhom = cv2.perspectiveTransform(cor, self._homo)[0]
            #Crop square
            crop = 3
            #Want to make trans image such size that all filled after transform
            top = int(max(corhom[0][1],corhom[2][1]))+crop
            bot = int(min(corhom[1][1],corhom[3][1]))-crop
            left = int(max(corhom[0][0],corhom[1][0]))+crop
            right = int(min(corhom[2][0], corhom[3][0]))-crop
              
            #Now we get image which we need to put at (top, left, right, bot)
            chomo = np.dot(np.matrix([[1,0,-left],[0,1,-top],[0,0,1]]), self._homo)
            s = (right-left, bot-top)
            if s[0] < 0 or s[1] < 0:
                return None, None, None
            transf_image = cv2.warpPerspective(src=image,M=chomo, dsize=s, flags=cv2.INTER_CUBIC)
            #print self._mask.dtype
            if self._mask.dtype == np.float or self._mask.dtype == np.uint8:
                inter = cv2.INTER_CUBIC
                print "cubic"
            else:
                inter = cv2.INTER_NEAREST
                print "nearest"

            transf_mask =  cv2.warpPerspective(src=self._mask,M=chomo, dsize=s, flags=inter)
            #print transf_mask.dtype
            shift = (self._data[0]+top, self._data[1]+left)
            return transf_image, shift, transf_mask

    def addBorders(self, shift, fshape, over=20):
        add = [0,0,0,0]
        h,w,_ = fshape
        hp,wp,_ = self._pano.shape
        print shift  
        #If negative move
        add[2] = max(-shift[1], 0)*over
        add[0] = max(-shift[0], 0)*over
        #If too small, make bigger
        add[3] = max(0, shift[1] + w - wp)*over
        add[1] = max(0, shift[0] + h - hp)*over
        nshift  = (shift[0]+add[0], shift[1]+add[2])
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
        if self._2D:
            self._data = nshift
        else:
            self._data = [self._data[0]+add[0], self._data[1]+add[2]]
       
        return nshift

    def add(self, shift, image, mask):
            h,w,_ = image.shape
            x1,x2,y1,y2 = shift[0],shift[0]+h, shift[1],shift[1]+w
            #Cant access object variables
            weights, pano, mask = self._weights, self._pano, mask
            code ="""
                for(int x = x1; x < x2; x++)
                    for(int y = y1; y < y2; y++)
                               """ + self.blendCode
            weave.inline(code,['x1','x2','y1','y2',
                               'mask','weights','pano','image'],
                         type_converters=weave.converters.blitz, 
                         support_code=self._support,
                         compiler="gcc")

    def blendNext(self, dataPoint):
        if dataPoint is None or dataPoint._homo is None:
            print "Image skiped, expect artifacts"
            self._stream.skipFrames(1)
        else:
            frame, shift, mask = self.getFrameMask(dataPoint)
            if frame is None:
                return False
            #if self._weights is not None:
            #    self.prevPano(image=self._weights.astype(np.uint16), window="weight")
            #self.prevPano(image=mask.astype(np.uint16), window="mask")
            #self.prevPano(image=self._pano, window="pano")

            nshift = self.addBorders(shift, frame.shape)
            #if nshift != shift:
            #    frame[:,:3]=[0,0,255]
            #    frame[:,-3:]=[0,0,255]
            #    frame[:3,:]=[0,0,255]
            #    frame[-3:,:]=[0,0,255]

            self.add(nshift, frame, mask) 




class BlendOverlayNearest(BlendOverlayCInline):
    _weights = True
    _mtype = np.int32
    _imtype = None
    maskCode = "{mask(x,y) = 1073741824 - (x-midx)*(x-midx) - (y-midy)*(y-midy);}"
    blendCode =  """
                        if ( weights(x,y) <= mask(x-x1,y-y1)){
                            weights(x,y) = mask(x-x1,y-y1);
                            pano(x,y,0) = image(x-x1,y-y1,0);
                            pano(x,y,1) = image(x-x1,y-y1,1);
                            pano(x,y,2) = image(x-x1,y-y1,2);
                            }
                        """
BlendRegister["Overlay nearest"] = (BlendOverlayNearest, None)

class BlendOverlayNearest2(BlendOverlayNearest):
    maskCode = """ {
                    int min = x;
                    if (y < min) min = y;
                    if (maxx-x < min) min = maxx-x;
                    if (maxy-y < min) min = maxy-y;
                    mask(x,y) = min+1;
                }"""

BlendRegister["Overlay nearest2"] = (BlendOverlayNearest2, None)

class BlendOverlayNearest3(BlendOverlayNearest):
    maskCode = """ {
                    int minx = x;
                    if (maxx-x < minx) minx = maxx-x;
                    int miny = y;
                    if (maxy-y < miny) miny = maxy-y;
                    mask(x,y) = miny+minx+1;
                }"""

BlendRegister["Overlay nearest3"] = (BlendOverlayNearest3, None)


class BlendOver(BlendOverlayCInline):
    _weights = False
    _mtype = np.uint8
    _imtype = None
    maskCode = "mask(x,y) = 255;"
    blendCode =  """
                        if (pano(x,y,0) != 0 && pano(x,y,1) != 0 && pano(x,y,2) !=0){
                            int m = mask(x-x1,y-y1);
                            pano(x,y,0) = (m * image(x-x1,y-y1,0) + pano(x,y,0)*(255-m))/255;
                            pano(x,y,1) = (m * image(x-x1,y-y1,1) + pano(x,y,1)*(255-m))/255;
                            pano(x,y,2) = (m * image(x-x1,y-y1,2) + pano(x,y,2)*(255-m))/255;
                        }else{
                            pano(x,y,0) = image(x-x1,y-y1,0);
                            pano(x,y,1) = image(x-x1,y-y1,1);
                            pano(x,y,2) = image(x-x1,y-y1,2);   
                        }
                        """

BlendRegister["Overlay"] = (BlendOver, None)

class BlendOverHalf(BlendOver):
    maskCode = "mask(x,y) = 128;"

BlendRegister["Overlay half"] = (BlendOverHalf, None)

class BlendAvarage(BlendOverlayCInline):
    _weights = True
    _mtype = np.int32
    _imtype = np.int32
    maskCode = "mask(x,y) = 1;"
    blendCode =  """    {
                        int m = mask(x-x1,y-y1);
                        weights(x,y) += m;
                        pano(x,y,0) += image(x-x1,y-y1,0)*m;
                        pano(x,y,1) += image(x-x1,y-y1,1)*m;
                        pano(x,y,2) += image(x-x1,y-y1,2)*m;
                        }
                        """
    def getPano(self):
        #we need to normalise
        retpano = self._pano.copy()
        retpano[:,:,0] /= self._weights
        retpano[:,:,1] /= self._weights
        retpano[:,:,2] /= self._weights
        #print retpano.max()
        return retpano.astype(np.uint8)

BlendRegister["Avarage"] = (BlendAvarage, None)

class BlendGrassFire(BlendAvarage):
    maskCode = """ {
                    int min = x;
                    if (y < min) min = y;
                    if (maxx-x < min) min = maxx-x;
                    if (maxy-y < min) min = maxy-y;
                    mask(x,y) = min+1;
                }"""

BlendRegister["Avarage grassfire"] = (BlendGrassFire, None)

class BlendGrassFire(BlendAvarage):
    maskCode = """ {
                    int min = x;
                    if (y < min) min = y;
                    if (maxx-x < min) min = maxx-x;
                    if (maxy-y < min) min = maxy-y;
                    mask(x,y) = (long long int)(min*min)/128 + 1;
                }"""

BlendRegister["Avarage grassfire pow2"] = (BlendGrassFire, None)

