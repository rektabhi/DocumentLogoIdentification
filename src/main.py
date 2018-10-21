# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:16:07 2018

@author: Abhishek Bansal
"""
import sys

from sklearn.externals import joblib

sys.path.append('..\..\LogoIdentification')
import numpy as np
from src import constants, utils
from src.loadImages import LoadImages
from src.HOG import HOG
from src.SIFT import SIFT
from src.SURF import SURF
from src.readLabels import readLabels
from src.classLabels import trainLabels, testLabels
import pickle


class Context:
    def __init__(self):
        self.images = LoadImages()
        self.hog = HOG()
        self.sift = SIFT()
        self.surf = SURF()
        self.stringLabels = readLabels()
        self.HOGMis = None
        self.SURFMis = None
        self.SIFTMis = None
        self.predFromProb = None
        self.predFromProbMis = None
        self.numOfLogosPerClass = None

    def loadData(self):
        self.images.loadTestImages()
        self.images.loadTrainImages()
        self.images.loadTrainLabels()
        self.images.loadTestLabels()

    def createModels(self):
        self.hog.matchHOGFeatures(
            self.images
        )
        self.sift.matchFeatures(
            self.images
        )
        self.surf.matchFeatures(
            self.images
        )

    def checkMis(self):
        self.HOGMis = utils.checkMispredictions(self.images.testLabels, self.hog.predictions)
        self.SURFMis = utils.checkMispredictions(self.images.testLabels, self.surf.predictions)
        self.SIFTMis = utils.checkMispredictions(self.images.testLabels, self.sift.predictions)

    def plotProbabilities(self):
        utils.plotSURFProb(self.surf.probability, self.images.testLabels)
        utils.plotSIFTProb(self.sift.probability, self.images.testLabels)
        utils.plotHOGProb(self.hog.probability, self.images.testLabels)

    def calculatePredFromProb(self):
        self.predFromProb = np.argmax(self.hog.probability, axis=1) + 1
        self.predFromProbMis = utils.checkMispredictions(self.images.testLabels, self.predFromProb)

    # Find optimum value of x and y which are used to find the better of the models
    def checkConfidenceOfDifferentClassifier(self):
        minimum = 20
        minx = None
        miny = None
        for y in np.linspace(0.1, 0.5, 40):
            for x in np.linspace(0, 10, 25):
                bestPrediction = np.zeros((constants.numTestImages,))
                for i in range(constants.numTestImages):
                    isSURFGood = False
                    isHOGGood = False
                    # TODO: Add SIFT Comparisons
                    isSIFTGood = False
                    ispredFromProbGood = False
                    if self.surf.probability[i] > x:
                        isSURFGood = True
                    if np.amax(self.hog.probability, axis=1)[i] > y:
                        ispredFromProbGood = True
                    if self.predFromProb[i] == self.surf.predictions[i]:
                        bestPrediction[i] = self.surf.predictions[i]
                    elif self.hog.predictions[i] == self.surf.predictions[i]:
                        bestPrediction[i] = self.surf.predictions[i]
                    elif ispredFromProbGood and not isSURFGood:
                        bestPrediction[i] = self.predFromProb[i]
                    elif not ispredFromProbGood and isSURFGood:
                        bestPrediction[i] = self.surf.predictions[i]
                    else:
                        bestPrediction[i] = self.predFromProb[i]

                bestMis = utils.checkMispredictions(testLabels, bestPrediction)
                print(x, y, utils.countMis(bestMis))
                if utils.countMis(bestMis) < minimum:
                    minimum = utils.countMis(bestMis)
                    minx = x
                    miny = y
        print(minx, miny, "Minimum: ", minimum)
        with open(constants.bestXY, 'wb+') as f:
            pickle.dump([minx, miny], f)

    def allMiss(self):
        count = 0
        for i in range(74):
            if self.HOGMis[i] and self.SURFMis[i] and self.predFromProbMis[i] and self.SIFTMis[i]:
                #    if bestMis[i]:
                count += 1
                utils.imshow(self.images.testImages[i])
        print(count)

    def loadModels(self):
        self.loadHOGModel()
        self.loadSIFTModel()
        self.loadSURFModel()
        self.numOfLogosPerClass = utils.numOfLogosPerClass(self.sift.trainLabels, constants.numLabels)
        with open(constants.bestXY, 'rb+') as f:
            self.x, self.y = pickle.load(f)

    def loadHOGModel(self):
        self.hog.loadHOGModel()

    def loadSIFTModel(self):
        self.sift.loadSIFTModel()

    def loadSURFModel(self):
        self.surf.loadSURFModel()




ctx = Context()
ctx.loadData()
ctx.createModels()
ctx.checkMis()
ctx.plotProbabilities()
ctx.calculatePredFromProb()
ctx.checkConfidenceOfDifferentClassifier()
ctx.allMiss()
