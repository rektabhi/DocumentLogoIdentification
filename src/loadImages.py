# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:02:20 2018

@author: Abhishek Bansal
"""

import cv2
from src.DetectNoise import detectNoise
from src import constants


def imbinarize(image):
    ret, imgf = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf


def loadTrainImages(dire=constants.trainImagesLocation, totalTrainImageCount=246):
    trainImages = []
    print("Loading Training Images")
    for index in range(100, totalTrainImageCount + 1):
        imgLoc = dire + str(index) + ".png"
        trainImages.append(cv2.imread(imgLoc, cv2.IMREAD_COLOR))
    print("Done!")
    print("Preprocessing Training Images")
    for index, img in enumerate(trainImages):
        trainImages[index] = preprocessImage(img)
    print("Done!")
    return trainImages


def loadTestImages(dire=constants.testImagesLocation, totalTestIamges=73):
    testImages = []
    print("Loading Test Images")
    for index in range(0, totalTestIamges + 1):
        imgLoc = dire + str(index) + ".png"
        testImages.append(cv2.imread(imgLoc, cv2.IMREAD_COLOR))
    print("Done!")
    print("Preprocessing Test Images")
    for index, img in enumerate(testImages):
        testImages[index] = preprocessImage(img)
    print("Done!")
    return testImages


def removeSaltAndPepperNoise(image):
    return cv2.medianBlur(image, 3)


def preprocessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = imbinarize(img)
    if detectNoise(img) > constants.saltAndPepperThreshold:
        img = removeSaltAndPepperNoise(img)
    return img