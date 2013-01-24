import cv2
import math

def focalFromHomography(H):
    h = H.flatten()
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2
    if v1 < v2:
        v2, v1 = v1, v2
    if v1 > 0 and v2 > 0:
        if abs(d1) > abs(d2):
            f1 = math.sqrt(v1)
        else:
            f1 = math.sqrt(v2)
    elif v1 > 0:
        f1 = math.sqrt(v1)
    else:
        f1 = None;

    d1 = h[0] * h[3] + h[1] * h[4];
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
    v1 = -h[2] * h[5] / d1;
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
    if v1 < v2:
        v2, v1 = v1, v2
    if v1 > 0 and v2 > 0:
        if abs(d1) > abs(d2):
            f0 = math.sqrt(v1)
        else:
            f0 = math.sqrt(v2)
    elif v1 > 0:
        f0 = math.sqrt(v1)
    else:
        f0 = None;

    return (f0, f1)

