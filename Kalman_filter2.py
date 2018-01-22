import numpy as np
import random
import cv2
import time
import math
import geompreds
from scipy import interpolate



MEASMATRIX = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
TRANSMATRIX = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
NOISECOV = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
kalman = [cv2.KalmanFilter(4, 2) for i in range(15)]
for kal in kalman:
    kal.measurementMatrix = MEASMATRIX
    kal.transitionMatrix = TRANSMATRIX
    kal.processNoiseCov = NOISECOV

prevpts0= [(0,0) for p in range(15)]
prevpts = [(0,0) for p in range(15)]
estimation = [(0, 0) for o in range(15)]
polate = interpolate.interp1d([0,0,0],[0,0,0], bounds_error=False)

def CCW(a,b,c):
    if geompreds.orient2d(a,b,c)>0:
        return True
    else: return False


def intersect(a,b,c,d):
    if CCW(a,c,d)==CCW(b,c,d):
        return False
    elif CCW(a,b,c)==CCW(a,b,d):
        return False
    else:
        return True

def convertKalmanArray(list1):
    temp = []
    for x, y in list1:
        temp.append(np.array([[np.float32(x)], [np.float32(y)]]))
    return temp

def updateKalman(t1, b1, b2, pts):
    global prevpts,prevpts0
    # prevpts0 = prevpts
    # prevpts = pts
    # for p in range(len(pts)):
    #     AM = vectAB(b1, pts[p])
    #     AB = vectAB(b1, t1)
    #     AD = vectAB(b1, b2)
    #     if not withinRegion(AM, AB, AD):
    #         pts[p] = t1
    pts2 = convertKalmanArray(pts)
    for x in range(len(kalman)):
        kalman[x].correct(pts2[x])


def vectAB(p1, p2):
    x2, y2 = p2
    x1, y1 = p1
    return x2 - x1, y2 - y1

def crossProd(v1, v2):
    a = np.array(v1)
    b = np.array(v2)
    return np.matmul(a,b )


def scalarMult(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    return (x1 * x2) + (y1 * y2)

# def intersect(p1, p2, p3, p4):
#     x1,y1 = p1
#     x2,y2 = p2
#     x3,y3 = p3
#     x4,y4 = p4
#     if max(x1, x2) < min(x3, x4) or max(y1, y2) < min (y3, y4):
#         return False
#     else:
#         return True

def nearestSide(t1, t2, b1, b2, prev, pt):

    if intersect(t1, t2, prev, pt):
        # print "segment "+ str(t1) + "and" + str(t1) + " intersect with " + str(prev)+ " and " + str(pt)
        return t1, t2
    elif intersect(t2, b2, prev, pt):
        # print "segment " + str(t2) + "and" + str(b2) + " intersect with " + str(prev) + " and " + str(pt)
        return t2, b2
    elif intersect(t1, b1, prev, pt):
        # print "segment " + str(t1) + "and" + str(b1) + " intersect with " + str(prev) + " and " + str(pt)
        return t1, b1
    elif intersect(b1, b2, prev, pt):
        # print "segment " + str(b1) + "and" + str(b2) + " intersect with " + str(prev) + " and " + str(pt)
        return b1, b2
    else:
        # print "segment " + str(b1) + "and" + str(b2) + " intersect with " + str(prev) + " and " + str(pt)
        return b1, b2

def withinRegion(AM, AB, AD):
    if (0 < scalarMult(AM, AB)) and (scalarMult(AM, AB) < scalarMult(AB, AB)) and (0 < scalarMult(AM, AD)) and (
                scalarMult(AM, AD) < scalarMult(AD, AD)):
        return True
    else:
        return False

def splinePred(kal, pts, hasROI):
    global polate, prevpts0,prevpts
    tempPrev = prevpts0
    prevpts0 = prevpts
    prevpts = pts
    temp = []
    for i in range(len(prevpts)):
        x1, y1 = prevpts0[i]
        x2, y2 = prevpts[i]
        x3, y3 = pts[i]
        x4, y4 = kal[i]
        xx = np.asarray([x1, x2, x3])
        yy = np.asarray([y1, y2, y3])
        if hasROI:
            try:
                polate = interpolate.interp1d(xx, yy, bounds_error=False)
                newY = int(polate(x4))
                temp.append((x4,newY))
            except:
                prevpts = prevpts0
                prevpts0 = tempPrev
                temp.append(pts[i])
        else:
            try:
                newY = int(polate(x4))
                temp.append((x4, newY))
            except:
                prevpts = prevpts0
                prevpts0 = tempPrev
                temp.append(pts[i])
    # if np.isnan(c):
    #     g = int(y3 + (y1+y2+y3)/3)
    #     return g
    return temp


def restrictEst(t1, t2, b1, b2, pt, index):
    AM = vectAB(b1, pt)
    AB = vectAB(b1, t1)
    AD = vectAB(b1, b2)
    if withinRegion(AM, AB, AD):
        return pt
        # x, y = pt
        # if x < 0 or x > 800 or y < 0 or y > 0:
        #     if x < 0:
        #         x = 0
        #     elif x > 800:
        #         x = 799
        #     if y < 0:
        #         y = 0
        #     elif y > 450:
        #         y = 449
        #     pt = (x, y)
        # return pt
    else:
        # print prevpts[index]
        # print estimation[index]
        # print nearestSide(t1, t2, b1, b2, prevpts[index], estimation[index])
        s1, s2 = nearestSide(t1, t2, b1, b2, prevpts[index], estimation[index])
        return intersection(s1, s2, prevpts[index], estimation[index])

def intersection(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001
    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;
    x2, y2 = pt2
    dx1 = x2 - x1;
    dy1 = y2 - y1
    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;
    xB, yB = ptB;
    dx = xB - x;
    dy = yB - y;

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (-1, -1)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (int(xi), int(yi))

def kalman_filter (s1, s2, s3, s4, pts, hasROI):
    global previousBlurr, prevpts

    updateKalman(s1, s3, s4, pts)                  # update with current reading

    prevpts = pts
    est = estimate(s1, s2, s3, s4)                   # get new points if the face is blurred
    spline = splinePred(est, pts, hasROI)
    return spline

def estimate(s1, s2, s3, s4):
    global estimation
    for i in range(len(prevpts)):
        tp = kalman[i].predict()
        x,y = int(tp[0]), int(tp[1])
        tp = (int(tp[0]), int(tp[1]))
        estimation[i] = restrictEst(s1, s2, s3, s4, tp, i)
        # for xx, yy in estimation:
        #     if xx>800 or yy>450:
        #         print "+=+=+=+=+==+=+=+=+"
        #         print s1, s2, s3, s4
        #         break
        # x, y = estimation[i]
        # if y < 0 or x < 0:
        #     print x, y
        #     print "==============================================================="
        #     print s1, s2, s3, s4, i
    return estimation
