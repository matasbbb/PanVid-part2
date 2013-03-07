import pickle
import os
class PredictSave(object):
    def __init__(self, path="tests/samples/data.ser"):
        self._path = path
        if os.path.exists(self._path):
            f = open(self._path, "rb")
            self._data = pickle.load(f)
            f.close()
            print "Loaded data for:"
            for d in self._data.keys():
                print d
        else:
            self._data = {}

    def getData(self, ident):
        if not self._data.has_key(ident):
            return None
        else:
            return self._data[ident]

    def setData(self, ident, data):
        self._data[ident] = data
        f = open(self._path, "wb")
        pickle.dump(self._data,f)
        f.close()




