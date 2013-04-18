import cv2
import numpy as np

class FixHomo():
    def __init__(self, datapoints):
        self.datapoints = datapoints
        self.dp = datapoints
    def track_point(self, x,y, points=None, skip=True):
        if points is None:
            points = self.datapoints
        diff_path = []
        homo = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
        for p in points:
            if p._homo is not None:
                homo = np.dot(p._homo, homo)
                cor = np.array([[x,y]],dtype='float32')
                cor = np.array([cor])
                corhom = cv2.perspectiveTransform(cor, homo)[0]
                #Intrested in diff!
                diff_path.append((corhom[0][0] - x, corhom[0][0] - y))
            elif not skip:
                diff_path.append((0,0))
        path = [[x,y]]
        now = [x,y]
        for p in diff_path:
            path.append((now[0]+p[0],now[1]+p[1]))
            now = path[-1]
        return diff_path, path
 
    def track_point2D(self, x,y, points=None, skip=True):
        if points is None:
            points = self.datapoints
        diff_path = []
        for p in points:
            if p._homo is not None:
                diff_path.append((p._homo[1][2], p._homo[0][2]))
            elif not skip:
                diff_path.append((0,0))
        path = [[x,y]]
        now = [x,y]
        for p in diff_path:
            path.append((now[0]+p[0],now[1]+p[1]))
            now = path[-1]
        return diff_path, path
   
    def _applyHomo(self, homo, points, rev=True):
        new = []
        for p in points:
            if p._homo is not None:
                if rev:
                    p._homo= np.dot(homo, p._homo)
                else:
                    p._homo= np.dot(p._homo, homo)
            new.append(p)
        return new

    def linealign(self,(x1,y1),(x2,y2), points=None):
        if points is None:
            points = self.datapoints

        homo = np.matrix([[1,0,(y2-y1)/len(points)],[0,1,(x2-x1)/len(points)],[0,0,1]])
        return self._applyHomo(homo, points)
        
    def linealign2D(self,(x1,y1),(x2,y2), points=None):
         
        return        
