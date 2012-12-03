import unittest
from panvid.generatePath import *
from panvid.generateVideo import *
import cv2


class TestPathGenerator(unittest.TestCase):
    def test_sweapPath(self):
        frames = 131
        gen = PathGenerator(frames, None, (1280,720), (100,100))
        path = gen.getSweapPath(100, False)
        self.assertEqual(len(path), frames)


class TestPathFilters(unittest.TestCase):
    def test_applyShake(self):
        frames = 131
        gen = PathGenerator(frames, None, (1280,720), (100,100))
        path = gen.getSweapPath(100, False)
        randompath = PathFilters(path).applyShake().getPath()
        self.assertEqual(len(randompath), frames)
        equal = 0
        for p1, p2 in zip(path, randompath):
            if p1 == p2:
                equal += 1
        self.assertTrue(equal < (frames/10))

    def test_applSppedChange(self):
        frames = 131
        gen = PathGenerator(frames, None, (1280,720), (100,100))
        path = gen.getSweapPath(100, False)
        randompath = PathFilters(path).applySpeedChange().getPath()
        self.assertEqual(len(randompath), frames)
        equal = 0
        for p1, p2 in zip(path, randompath):
            if p1 == p2:
                equal += 1
        self.assertTrue(equal < (frames/10))

class TestVideoSimpleGeneratir(unittest.TestCase):
    def test_save(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        gen = PathGenerator(300, img, None, frame_size)
        path = gen.getSweapPath(2000, False)

        shpath = PathFilters(path).applyShake().getPath()
        vidgen = VideoSimpleGenerator(shpath, img)
        vidgen.save("/tmp/test_shake.avi", frame_size, fps=30)

        sppath = PathFilters(path).applySpeedChange().getPath()
        vidgen = VideoSimpleGenerator(sppath, img)
        vidgen.save("/tmp/test_speed.avi", frame_size, fps=30)

        spshpath = PathFilters(path).applySpeedChange().applyShake().getPath()
        vidgen = VideoSimpleGenerator(spshpath, img)
        vidgen.save("/tmp/test_speed_shake.avi", frame_size, fps=30)

class TestVideoEffects(unittest.TestCase):
    def test_noise(self):
        frame_size = (1000,1000)
        img = cv2.imread("tests/samples/IMG_8686.JPG")
        path = [(2000,2000)] * 30
        filt = VideoEffects()
        filt.noise()
        vidgen = VideoSimpleGenerator(path, img)
        vidgen.save("/tmp/test_noise.avi", frame_size, fps=30, filt=filt)

if __name__ == '__main__':
    unittest.main()
