# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:16:07 2018

@author: Abhishek Bansal
"""
import numpy as np
from src import loadImages, SIFT, constants, HOG, SURF, utils

from src.readLabels import readLabels
from src.classLabels import y as train_labels, y2 as test_labels
import pickle

trainImages = loadImages.loadTrainImages()
testImages = loadImages.loadTestImages()
stringLabels = readLabels()

HOGPredictions, HOGProb = HOG.matchHOGFeatures(trainImages, testImages, train_labels, test_labels)
SURFPredictions, SURFProb = SURF.matchFeatures(trainImages, testImages, train_labels, test_labels)
SIFTPredictions, SIFTProb = SIFT.matchFeatures(trainImages, testImages, train_labels, test_labels)

HOGMis = utils.checkMispredictions(test_labels, HOGPredictions)
SURFMis = utils.checkMispredictions(test_labels, SURFPredictions)
SIFTMis = utils.checkMispredictions(test_labels, SIFTPredictions)

# print("HOGMis: ", utils.countMis(HOGMis))
# print("SURFMis: ", utils.countMis(SURFMis))
predFromProb = np.argmax(HOGProb, axis=1) + 1
predFromProbMis = utils.checkMispredictions(test_labels, predFromProb)
# print(utils.countMis(predFromProbMis))
print(utils.countMis(SIFTMis))
# print(utils.countMis(utils.checkMispredictions(HOGPredictions, predFromProb)))

utils.plotHOGProb(HOGProb, test_labels)
utils.plotSURFProb(SURFProb, test_labels)


# Find optimum value of x and y which are used to find the better of the models
def confidenceCheck():
    minimum = 20
    for y in np.linspace(0.1, 0.5, 40):
        for x in np.linspace(0, 10, 25):
            bestPrediction = np.zeros((constants.numTestImages,))
            for i in range(constants.numTestImages):
                isSURFGood = False
                isHOGGood = False
                # TODO: Add SIFT Comparisons
                isSIFTGood = False
                ispredFromProbGood = False
                if SURFProb[i] > x:
                    isSURFGood = True
                if np.amax(HOGProb, axis=1)[i] > y:
                    ispredFromProbGood = True
                if predFromProb[i] == SURFPredictions[i]:
                    bestPrediction[i] = SURFPredictions[i]
                elif HOGPredictions[i] == SURFPredictions[i]:
                    bestPrediction[i] = SURFPredictions[i]
                elif ispredFromProbGood and not isSURFGood:
                    bestPrediction[i] = predFromProb[i]
                elif not ispredFromProbGood and isSURFGood:
                    bestPrediction[i] = SURFPredictions[i]
                else:
                    bestPrediction[i] = predFromProb[i]

            bestMis = utils.checkMispredictions(test_labels, bestPrediction)
            print(x, y, utils.countMis(bestMis))
            if utils.countMis(bestMis) < minimum:
                minimum = utils.countMis(bestMis)
                minx = x
                miny = y
    print(minx, miny, "Minimum: ", minimum)
    with open(constants.bestXY, 'wb+') as f:
        pickle.dump([minx, miny], f)


confidenceCheck()
count = 0
for i in range(74):
    if HOGMis[i] and SURFMis[i] and predFromProbMis[i] and SIFTMis[i]:
        #    if bestMis[i]:
        count += 1
        utils.imshow(testImages[i])
print(count)
