# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:46:33 2018

@author: Abhishek Bansal
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src import constants


def imshow(image):
    cv2.imshow('image', image)
    cv2.waitKey(constants.waitTimeImage)
    cv2.destroyAllWindows()


def numOfLogosPerClass(labels, n):
    numLabels = np.zeros((n,))
    for label in labels:
        numLabels[label - 1] += 1
    return numLabels


def checkMispredictions(actualLabels, predictedLabels):
    actualLabels = np.array(actualLabels)
    predictedLabels = np.array(predictedLabels)
    (num,) = np.shape(actualLabels)
    mispredictions = np.zeros((num,), dtype=np.bool)
    for index in range(num):
        if actualLabels[index] == predictedLabels[index]:
            mispredictions[index] = False
        else:
            mispredictions[index] = True
    return mispredictions


def countMis(predictedLabels):
    count = 0
    for result in predictedLabels:
        if result:
            count += 1
    return count


def plotHOGProb(HOGProb, actualLabels):
    predFromProb = np.argmax(HOGProb, axis=1) + 1
    actualLabels = np.array(actualLabels)
    maxProbs = np.amax(HOGProb, axis=1)
    plt.plot(maxProbs)


def plotSURFProb(SURFProb, actualLabels):
    actualLabels = np.array(actualLabels)
    plt.plot(SURFProb)


def plotSIFTProb(SIFTProb, actualLabels):
    actualLabels = np.array(actualLabels)
    plt.plot(SIFTProb)


def rgb2gray(image):
    if image is None:
        return "Image not found"
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        return image


def imbinarize(image, mode=constants.ADAPTIVE_THRESHOLD):
    if mode is constants.ADAPTIVE_THRESHOLD:
        imgf = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        ret, imgf = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return imgf


def imcomplement(image):
    return cv2.bitwise_not(image)


def resize(image):
    height, width = image.shape[:2]
    max_height = constants.maxHeight
    max_width = constants.maxWidth
    if height > max_height or width > max_width:
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return image

