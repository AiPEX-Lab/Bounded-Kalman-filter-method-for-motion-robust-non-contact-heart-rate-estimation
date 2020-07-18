# import the necessary packages
from __future__ import division
from common import analyze_channels
from common import parabolic as parabolic
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import numpy.fft as fft
import time
import copy
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
from scipy import signal
import scipy as scipy
from Kalman_filter2 import *
import pandas as pd
from scipy.signal import butter, lfilter
from scipy import linalg as LA
from random import randint
from numpy.fft import rfft
from numpy import argmax, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
import sys
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA, PCA
from multiprocessing import Process, Lock
previous = 0
dataset = [0 for i in range(10)]


def openGraph(lock):
    from matplotlib import animation
    def update_line(num, data, line):
        global previous
        # if os._exists('bpmTemp.txt'):
        lock.acquire()
        textFile = open("bpmTemp.txt", 'r')
        bpm = float(textFile.read())
        previous = bpm
        textFile.close()
        lock.release()

        data.append(previous)
        plt.clf()
        plt.plot(data, 'r-')
        x = plt.gca()
        x.set_xlabel('Time (seconds)', fontsize=35)
        x.set_ylabel('B.P.M.', fontsize=35)

        fig1.suptitle('Heart Rate(B.P.M)\n' + str(previous), fontsize=35, fontweight='bold')
        # line.set_ydata(data)
        plt.xlim([num - 9, num + 1])
        return line,

    fig1 = plt.figure(figsize=(10,10))
    bpmtext = fig1.suptitle('Heart Rate(B.P.M)\n' + str(previous), fontsize=35, fontweight='bold')
    # bpmtext = plt.figtext(0.9, 0.5, 'Heart Rate\n(B.P.M)\n' + str(previous), fontdict=None, \
    #                       fontsize=16, fontweight='bold', position=(1, 0.5))
    # fig1.suptitle('BMP graph', fontsize=14, fontweight='bold')
    x = plt.gca()
    x.set_xlabel('Time (seconds)', fontsize=35)
    x.set_ylabel('B.P.M.', fontsize=35)
    data = [previous for i in xrange(10)]
    l, = plt.plot(data, 'r-')
    plt.ylim(40, 180)
    line_ani = animation.FuncAnimation(fig1, update_line, fargs=(data, l), interval=1000, blit=False)
    plt.show()
#Estimate frequency from peak of FFT
#returns calculated frequency
def freq_from_fft(sig, fs):
    
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    fe = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(fe))  # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(fe)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)



#creates a butter pass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

#creates a butter pass filter by cutting off certain frequency
#returns filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian
#retruns variance
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

#finds mean of a lsit
#retruns the mean
def findvarmean(alist):
    b = 0
    for a in alist:
        b = b+variance_of_laplacian(a)
    return b/len(alist)


#cheacks particular pixel difference from two lists consisting of pexel location
#returns a boolean depending on the pixel distance
def checkpixeldiff(a,b):
    if b[0] == 0:
        return True
    else:
        c1 = abs(a[0] - b[0])
        c2 = abs(a[1] - b[1])
        c3 = abs(a[2] - b[2])
        c4 = abs(a[3] - b[3])
        if c1<50 and c2<50 and c3<50 and c4<50:
            return True
        else:
            return False

#converta frame to HSV
#retruns HSV frame
def CoverttoHSV(image1):
    HSV = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    return HSV




def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


#main program
# def main(lock2):
def main():

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Input a video")
    args = vars(ap.parse_args())

    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # if the video path was not supplied, grab the reference to the
    # camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])

    # setup the mouse callback
    cv2.namedWindow("frame")
    # cv2.setMouseCallback("frame", selectROI)

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # initialize Variables
    frame, nextgray, prevgray, currROI1, currROI2, currROI3, normalizedrectangle1, normalizedrectangle2, normalizedrectangle3, nextframe, previousFrame, threshold, listtocheck, roiBox, tao = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    meanlist, listoftimes, roiPts, HistoryList, prevlistforehead, prevlistfleftface, prevlistrightface, prevallpoints = [], [], [], [], [], [], [], []
    a11, a22, c11, b11, b22, c22, prevframecounter, listcounter, firstframe, initframecounter, firsttimecount, cutlow, frameno, fcount = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    inputMode, resetloop, writetext, HRready = False, False, True, False
    enterif, notmoving, first, thritysecs = True, True, True, True
    detectioncount1, detectioncount2 = 0, 0
    countdowntime = 1
    result = []
    topleft, bottomleft, topright, bottomright = (0, 0), (0, 0), (0, 0), (0, 0)
    cuthigh = 10
    currentcount = 5
    ptsss = [(0, 0) for i in range(15)]
    ctd = "Calibrating"

    # reading the next frame frame and starting time
    nextframe = camera.read()
    start_time = time.time()
    # creating a file for writing
    textFilee = open('test.txt', 'w') #this textfile records all heart rate values until the end of video or 'q' is pressed
    # saving the frames as .mp4 video
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('original_vid.mp4', fourcc, 25.0, (480, 640))
    # keep looping over the frames
    while True:
        # setting face not detected to begin with
        foundlandmark = False

        # resetting all the variable to initialed value when resetloop is envoked
        if resetloop == True:
            listtocheck = None
            firstframe = 1
            initframecounter = 0
            firsttimecount = 0
            countdowntime = 1
            resetloop = False
            enterif = True
            notmoving = True
            first = True

        # grab the current frame
        (grabbed, frame) = camera.read()
        # resizing the frame
        frame =  cv2.resize(frame, (640, 480))
        # copying the frame
        HSVframe = copy.copy(frame)
        
        # setting precious frame and next frame
        prevframe = copy.copy(nextframe)
        nextframe = copy.copy(frame)
        # check to see if we have reached the end of the
        # video
        if not grabbed:
            break

        # if the see if the ROI has been computed
        if roiBox is not None:
            # convert the current frame to the HSV color space
            # and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)

            # Now convolute with circular disc
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cv2.filter2D(backProj, -1, disc, backProj)

            # threshold and binary AND
            ret, thresh = cv2.threshold(backProj, 50, 255, 0)
            thresh = cv2.merge((thresh, thresh, thresh))
            skinMask = cv2.erode(thresh, disc, iterations=2)
            skinMask = cv2.dilate(thresh, disc, iterations=2)

            # blur the mask to help remove noise, then apply the
            # mask to the frame
            skinMask = cv2.GaussianBlur(thresh, (3, 3), 0)
            res = cv2.bitwise_and(frame, skinMask)
            fcount = fcount + 1
            # draw a box on the indentified density

            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # converting frame to grayscale
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)
            # declaring variables
            ixc, iyc, a1x, a1y, a2x, a2y, a3x, a3y, a4x, a4y, a5lx, a5y, b1x, b1y, b2x, b2y, b3x, b3y, b4x, b4y, b5y, b5x, c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, c5x, c5y, d1x, d1y, e1x, e1y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                counter = 0

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image

                for (x, y) in shape:

                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(counter), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    # setting face as found
                    foundlandmark = True

                    # saving particular face landmarks for the ROI box
                    if counter == 21:
                        a1x = x
                        a1y = y / 1.3
                    if counter == 22:
                        a2x = x
                        a2y = y
                    if counter == 27:
                        a3x = x
                        a3y = y
                    if counter == 8:
                        a4x = x
                        a4y = y
                    if counter == 23:
                        a5x = x
                        a5y = y

                    if counter == 17:
                        b1x = x
                        b1y = y * 1.2
                    if counter == 31:
                        b2x = x
                        b2y = y
                    if counter == 28:
                        b3x = x
                        b3y = y
                    if counter == 39:
                        b4x = x
                        b4y = y
                        ixc = (a1x + a2x) / 2.2
                        iyc = (a4y + a3y)

                    if counter == 26:
                        c1x = x
                        c1y = y / 1.2
                    if counter == 35:
                        c2x = x
                        c2y = y
                    if counter == 28:
                        c3x = x
                        c3y = y
                    if counter == 42:
                        c4x = x
                        c4y = y

                    if counter == 16:
                        d1x = x * 1.1
                        d1y = y

                    if counter == 0:
                        e1x = x / 1.15
                        e1y = y

                    counter = counter + 1

                firstframe = 0
                initframecounter = initframecounter + 1

            # co-ordinates for the rectangle
            listforehead = [int(a1x), int(a1y), a2x, a2y]
            listleftface = [int(b1x), int(b3y), b4x, b2y]
            listrightface = [int(c1x), int(c3y), c4x, c2y]

            cv2.rectangle(frame, (listforehead[0], listforehead[1]), (listforehead[2], listforehead[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (listleftface[0], listleftface[1]), (listleftface[2], listleftface[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (listrightface[0], listrightface[1]), (listrightface[2], listrightface[3]),
                          (255, 0, 0), 2)

            # converting the frame to HSV
            HSVframe = CoverttoHSV(HSVframe)

            # checkig if this is the first frame
            if firstframe == 0:
                if first == True:
                    listtocheck = listforehead
                    firstframe = 1
                    first = False

            # setting up intital frames to measure the Bluriness value
            # setting up the blurriness threshold from the frist 6 seconds
            # checking the following frames and comparing it to the bluriness value
            # sharpen the frames if needed depending on the blurriness mean
            if (initframecounter / 2) < 6:
                if enterif == True:
                    if countdowntime == 0:
                        HistoryList.append(gray)
                    notmoving = checkpixeldiff(listtocheck, listforehead)
                    if notmoving == False:
                        text = "You Moved. Starting countdown again"
                        resetloop = True
                        HistoryList = []
                        continue

            else:
                if enterif == True:
                    initframecounter = 0
                    if countdowntime == 0:
                        notmoving = checkpixeldiff(listtocheck, listforehead)
                        if notmoving == True:
                            enterif = False
                            continue
                        else:
                            resetloop = True
                            HistoryList = []

                            continue
                    countdowntime = countdowntime - 1

            if enterif == False:

                threshold = findvarmean(HistoryList)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < threshold:
                    HSVframe = cv2.bilateralFilter(HSVframe, 9, 75, 75)
                    gaussian = cv2.GaussianBlur(HSVframe, (9, 9), 10.0)
                    HSVframe = cv2.addWeighted(HSVframe, 1.5, gaussian, -0.5, 0, HSVframe)


                else:
                    HistoryList.pop(listcounter)
                    HistoryList.append(gray)
                    threshold = findvarmean(HistoryList)
                    listcounter = listcounter + 1
                    if listcounter == len(HistoryList):
                        listcounter = 0

            if enterif == True:
                currentcount = countdowntime
                if foundlandmark == False:
                    ctd = "Searching For Face"
                else:
                    ctd = "Calibrating"

                ctd2 = ctd
                cv2.putText(frame, ctd, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))

            # setting the previous ROI to cerrent ROI
            prevROI1 = currROI1
            prevROI2 = currROI2
            prevROI3 = currROI3

            # setting the current ROI to next ROI
            currROI1 = gray[listforehead[1]:listforehead[1] + 10, listforehead[0]:listforehead[0] + 10]
            currROI2 = gray[listleftface[1]:listleftface[1] + 10, listleftface[0]:listleftface[0] + 10]
            currROI3 = gray[listrightface[1]:listrightface[1] + 10, listrightface[0]:listrightface[0] + 10]

            pointsListRoi1, pointsListRoi2, pointsListRoi3, avelist1, avearray2, Normalizedlist = [], [], [], [], [], [
                8]

            # finding the middle points of the region of interest for Kalman Filter Calculation
            for x1 in range(1):
                a1 = int((listforehead[0] + listforehead[2]) / 2) + x1
                b1 = int((listleftface[0] + listleftface[2]) / 2) + x1
                c1 = int((listrightface[0] + listrightface[2]) / 2) + x1
                for y1 in range(5):
                    a2 = int((listforehead[1] + listforehead[3]) / 2) + y1
                    b2 = int((listleftface[1] + listleftface[3]) / 2) + y1
                    c2 = int((listrightface[1] + listrightface[3]) / 2) + y1

                    tup1 = (a1, a2)
                    tup2 = (b1, b2)
                    tup3 = (c1, c2)

                    pointsListRoi1.append(tup1)
                    pointsListRoi2.append(tup2)
                    pointsListRoi3.append(tup3)
                    allPoints = pointsListRoi1 + pointsListRoi2 + pointsListRoi3
                d = 0

            # if face is found
            if foundlandmark == True:
                # seeting the previous to current
                prevlistforehead = listforehead
                prevlistleftface = listleftface
                prevlistrightface = listrightface
                prevallpoints = allPoints
                topright = (pts[0][0], pts[0][1])
                bottomright = (pts[1][0], pts[1][1])
                topleft = (pts[3][0], pts[3][1])
                bottomleft = (pts[2][0], pts[2][1])

                # Passing the points to the Kalman filter
                ptsss = kalman_filter(topleft, bottomleft, topright, bottomright, allPoints, foundlandmark)

                # finding the length of the ROI
                a11 = int(abs(listforehead[0] - listforehead[2]) / 4)
                b11 = int(abs(listleftface[0] - listleftface[2]) / 4)
                c11 = int(abs(listrightface[0] - listrightface[2]) / 4)

                a22 = int(abs(listforehead[1] - listforehead[3]) / 4)
                b22 = int(abs(listleftface[1] - listleftface[3]) / 4)
                c22 = int(abs(listrightface[1] - listrightface[3]) / 4)

                ptsss2 = [ptsss[0], ptsss[5], ptsss[10]]

                # Finding the HSV value of the points of the ROI and storing it
                for xaxis in range(ptsss[0][0] - a11, ptsss[0][0] + a11):
                    for yaxis in range(ptsss[0][1] - a22, ptsss[0][1] + a22):
                        Normalizedlist.append(HSVframe[yaxis][xaxis][0])
                cv2.circle(frame, (ptsss[0][0], ptsss[0][1]), 8, (0, 0, 255), -1)

                for xaxis in range(ptsss[5][0] - b11, ptsss[5][0] + b11):
                    for yaxis in range(ptsss[5][1] - b22, ptsss[5][1] + b22):
                        Normalizedlist.append(HSVframe[yaxis][xaxis][0])
                cv2.circle(frame, (ptsss[5][0], ptsss[5][1]), 8, (0, 0, 255), -1)

                for xaxis in range(ptsss[10][0] - c11, ptsss[10][0] + c11):
                    for yaxis in range(ptsss[5][1] - c22, ptsss[5][1] + c22):
                        Normalizedlist.append(HSVframe[yaxis][xaxis][0])
                cv2.circle(frame, (ptsss[10][0], ptsss[10][1]), 8, (0, 0, 255), -1)

                avearray2 = np.asarray(Normalizedlist)
                # taking the mean of the ROi
                totalmean = int(np.mean(avearray2))

            else:

                # When face not found work with previous values
                # Passing the points to the Kalman filter
                ptsss = kalman_filter(topleft, bottomleft, topright, bottomright, ptsss, foundlandmark)
                ptsss2 = [ptsss[0], ptsss[5], ptsss[10]]

                # Finding the HSV value of the points of the ROI and storing it
                for xaxis in range(ptsss[0][0] - a11, ptsss[0][0] + a11):
                    for yaxis in range(ptsss[0][1] - a22, ptsss[0][1] + a22):
                        Normalizedlist.append(HSVframe[yaxis][xaxis][0])

                # cv2.circle(frame, (ptsss[0][0], ptsss[0][1]), 8, (0, 0, 255), -1)

                for xaxis in range(ptsss[5][0] - b11, ptsss[5][0] + b11):
                    for yaxis in range(ptsss[5][1] - b22, ptsss[5][1] + b22):
                        Normalizedlist.append(HSVframe[yaxis][xaxis][0])

                # cv2.circle(frame, (ptsss[5][0], ptsss[5][1]), 8, (0, 0, 255), -1)


                for xaxis in range(ptsss[10][0] - c11, ptsss[10][0] + c11):
                    for yaxis in range(ptsss[5][1] - c22, ptsss[5][1] + c22):
                        Normalizedlist.append(HSVframe[yaxis][xaxis][0])
                # cv2.circle(frame, (ptsss[10][0], ptsss[10][1]), 8, (0, 0, 255), -1)


                avearray2 = np.asarray(Normalizedlist)
                # Taking the mean of the ROI
                totalmean = int(np.mean(avearray2))

            # upldating frame number and storing the mean
            frameno = frameno + 1
            end_time = time.time() #record end time of program
            meanlist.append(totalmean)
            listoftimes.append(frameno)

            # converting to numpy array
            alll = np.asarray(meanlist)
            nptime = np.asarray(listoftimes)
            loop = False

            if foundlandmark == False:
                detectioncount1 = detectioncount1 + 1
            else:
                detectioncount1 = 0

            if detectioncount1 == 20:
                loop = True
                detectioncount1 = 0

            if foundlandmark == False and loop == True:
                roiPts = []
                roiBox = None

            # checking if enough samples are collected
            # Following is all the signal processing steps
            if len(alll) >= 125 + cutlow:     # fps = 30 frames/second, 300frames = 10 second window (30frames *10seconds)
                HRready = True
                global hr
                FPS =25.00
                WINDOW_TIME_SEC = 5
                WINDOW_SIZE = int(np.ceil(WINDOW_TIME_SEC * FPS))
                windowStart = len(alll) - WINDOW_SIZE
                window = alll[windowStart: windowStart + WINDOW_SIZE]
                window = np.asarray(window)
                # ica = FastICA(whiten=False)
                window = (window - np.mean(window, axis=0)) / np.std(window, axis=0)  # signal normalization
                window = np.reshape(window, (125, 1))
                # S = ica.fit_transform(window)  # ICA Part
                fs = FPS
                lowcut = 0.75
                highcut = 2.5
                detrend = scipy.signal.detrend(window)
                y = butter_bandpass_filter(detrend, lowcut, highcut, fs, order=5)
                powerSpec = np.abs(np.fft.fft(y, axis=0)) ** 2
                freqs = np.fft.fftfreq(window.shape[0], 1.0 / fs)
                MIN_HR_BPM = 45.0
                MAX_HR_BMP = 150.0
                MAX_HR_CHANGE = 2.0
                SEC_PER_MIN = 60
                maxPwrSrc = np.max(powerSpec, axis=1)
                validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
                validPwr = maxPwrSrc[validIdx]
                validFreqs = freqs[validIdx]
                maxPwrIdx = np.argmax(validPwr)
                hr = validFreqs[maxPwrIdx]
                cutlow = cutlow + FPS
                out6 = hr * 60
                result.append(out6)
                ave = np.asarray(result)
                out6 = int(np.mean(ave))

                global previous
                previous = out6
                # lock2.acquire() # uncomment this if graph is required
                textFile = open('bpmTemp.txt', 'w') #this textfile updates the graph if graph option is uncommented 
                textFile.write(str(out6))
                textFile.close()
                # lock2.release() # uncomment this if graph is required

                tao = str('%.2f' % (out6))

                ce = 'BPM: ' + tao
                if enterif == False:
                    cv2.putText(frame, ce, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
                if writetext == True:
                    textFilee.write(tao + '\n') #this textfile records all heart rate values until the end of video or 'q' is pressed

            else:
                if enterif == False:
                    if HRready == False:
                        if foundlandmark == False:
                            ctd = "Searching For Face"
                        else:
                            ctd = "Calibrating"
                        cv2.putText(frame, ctd, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))


                    else:
                        t = str('%.2f' % out6)
                        fe = 'BPM: ' + t
                        cv2.putText(frame, fe, (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        cv2.imshow("frame", frame)
        cv2.imwrite("frame.jpg",frame)
        # out.write(frame) #comment out if saving video output
        key = cv2.waitKey(1) & 0xFF

        # handle if the 'i' key is pressed, then go into ROI
        # selection mode
        if len(roiPts) < 4:
        #if len(roiPts) < 4:
            # indicate that we are in input mode and clone the
            # frame
            inputMode = True
            orig = frame.copy()

            # keep looping until 4 reference ROI points have
            # been selected; press any key to exit ROI selction
            # mode once 4 points have been selected
            while len(roiPts) < 4:
                foundface = False
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                top1 = (0, 0)
                top2 = (0, 0)
                bottom1 = (0, 0)
                bottom2 = (0, 0)
                for (x, y, w, h) in faces:
                    foundface = True
                    cv2.rectangle(frame, (int(x + 0.2 * w), y), (int(x + 0.8 * w), int(y + 0.78 * h)), (255, 0, 0), 2)
                    top1 = (int(x + 0.2 * w), y)
                    top2 = (int(x + 0.8 * w), y)
                    bottom1 = (int(x + 0.2 * w), int(y + 0.78 * h))
                    bottom2 = (int(x + 0.8 * w), int(y + 0.78 * h))

                cv2.imshow("frame", frame)
                # cv2.waitKey(0)

                if foundface == True:
                    roiPts.append(top1)
                    roiPts.append(top2)
                    roiPts.append(bottom1)
                    roiPts.append(bottom2)

            # determine the top-left and bottom-right points
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis=1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]

            # grab the ROI for the bounding box and convert it
            # to the HSV color space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

            # compute a HSV histogram for the ROI and store the
            # bounding box
            roiHist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])

        # if the 'q' key is pressed, stop the loop
        elif key == ord("q"):
            break
        # Press T for text file writing
        elif key == ord("t"):
            writetext = True

    # textFile.close()
    textFilee.close()
    # cleanup the camera and close any open windows
    camera.release()
    # out.release() #comment out if saving video output
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    ## If you want a real-time graph along with HR estimation, comment the line above and line 165, uncomment all lines below and line 164, 624, 628
    # lock = Lock()
    # lock2 = Lock()
    # p1 = Process(target=openGraph, args=((lock),))
    # p2 = Process(target=main, args=((lock2),))
    # p2.start()
    # p1.start()
    # p2.join()
    # p1.join()

