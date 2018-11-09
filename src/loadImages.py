# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:02:20 2018

@author: Abhishek Bansal
"""

import cv2
from src import constants, utils
from src.DetectNoise import detectNoise
from src.classLabels import trainLabels, testLabels


class LoadImages:
    
    def __init__(self):
        self.trainImages = None
        self.trainLabels = None
        self.testImages = None
        self.testLabels = None
        self.numOfLogosPerClass = None

    def imbinarize(self, image):
        ret, imgf = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return imgf

    def loadTrainImages(self, dire=constants.trainImagesLocation, totalTrainImageCount=constants.numTrainImages):
        self.trainImages = []
        print("Loading Training Images")
        for index in range(100, totalTrainImageCount):
            imgLoc = dire + str(index) + ".png"
            self.trainImages.append(cv2.imread(imgLoc, cv2.IMREAD_COLOR))
        print("Done!")
        print("Pre-processing Training Images")
        for index, img in enumerate(self.trainImages):
            self.trainImages[index] = self.preprocessImage(img)
        print("Done!")

    def loadTestImages(self, dire=constants.testImagesLocation, totalTestIamgesCount=constants.numTestImages):
        self.testImages = []
        print("Loading Test Images")
        for index in range(0, totalTestIamgesCount):
            imgLoc = dire + str(index) + ".png"
            self.testImages.append(cv2.imread(imgLoc, cv2.IMREAD_COLOR))
        print("Done!")
        print("Preprocessing Test Images")
        for index, img in enumerate(self.testImages):
            self.testImages[index] = self.preprocessImage(img)
        print("Done!")

    def loadTrainLabels(self):
        self.trainLabels = trainLabels

    def loadTestLabels(self):
        self.testLabels = testLabels

    def removeSaltAndPepperNoise(self, image):
        return cv2.medianBlur(image, 3)

    def preprocessImage(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = utils.imbinarize(img)
        if detectNoise(img) > constants.saltAndPepperThreshold:
            img = self.removeSaltAndPepperNoise(img)
        return img

    def loadNumOfLogosPerClass(self):
        self.numOfLogosPerClass = utils.numOfLogosPerClass(self.trainLabels, constants.numLabels)
