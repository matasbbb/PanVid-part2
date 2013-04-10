import pickle
import os
class PredictSave(object):
    def __init__(self, path="tests/samples/data.ser"):
        self._path = path
        if os.path.exists(self._path):
            f = open(self._path, "rb")
            self._data = pickle.load(f)
            f.close()
            #print "Loaded data for:"
            #for d in self._data.keys():
            #    print d
        else:
            self._data = {}

    def getData(self, ident, skip=0):
        if not self._data.has_key(ident):
            return None
        else:
            data =  self._data[ident]
            if skip != 0:
                #we need to join...
                ndata = []
                index = 1
                joined = 0
                homo = data[0]
                while index < len(data):
                    if joined < skip:
                        homo = homo*data[index] 
                        joined += 1
                    else:
                        ndata.append(homo)
                        homo = data[index]
                        joined = 0
                    index += 1
                #We dont append non compleate one!
                return ndata
            return data
    def setData(self, ident, data):
        self._data[ident] = data
        f = open(self._path, "wb")
        pickle.dump(self._data,f)
        f.close()




