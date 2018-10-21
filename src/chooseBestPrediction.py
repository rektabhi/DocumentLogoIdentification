# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 00:42:29 2018

@author: Abhishek Bansal
"""
import numpy as np


def chooseBestPrediction(model, predictedValues):
    isSURFGood = False
    isHOGGood = False
    # TODO: Add SIFT Comparisons
    isSIFTGood = False
    ispredFromProbGood = False
    predFromProb = np.argmax(predictedValues["HOGProb"], axis=1) + 1
    if predictedValues["predictedSURFConfidence"] > model.x:
        isSURFGood = True
    if np.amax(predictedValues["HOGProb"], axis=1) > model.y:
        ispredFromProbGood = True
    if predFromProb == predictedValues["predictedSURFClass"]:
        bestPrediction = predictedValues["predictedSURFClass"]
    elif predictedValues["predictedHOGClass"] == predictedValues["predictedSURFClass"]:
        bestPrediction = predictedValues["predictedSURFClass"]
    elif ispredFromProbGood and not isSURFGood:
        bestPrediction = predFromProb
    elif not ispredFromProbGood and isSURFGood:
        bestPrediction = predictedValues["predictedSURFClass"]
    else:
        bestPrediction = predFromProb
    return bestPrediction

# def createBestPrediction():
#     for y in np.linspace(0.1, 0.5, 40):
#         for x in np.linspace(0, 10, 25):
#             bestPrediction = np.zeros((constants.numTestImages,))
#             for i in range(constants.numTestImages):
#                 isSURFGood = False
#                 isHOGGood = False
#                 # TODO: Add SIFT Comparisons
#                 isSIFTGood = False
#                 ispredFromProbGood = False
#                 if SURFProb[i] > x:
#                     isSURFGood = True
#                 if np.amax(HOGProb, axis=1)[i] > y:
#                     ispredFromProbGood = True
#                 if predFromProb[i] == SURFPredictions[i]:
#                     bestPrediction[i] = SURFPredictions[i]
#                 elif HOGPredictions[i] == SURFPredictions[i]:
#                     bestPrediction[i] = SURFPredictions[i]
#                 elif ispredFromProbGood and not isSURFGood:
#                     bestPrediction[i] = predFromProb[i]
#                 elif not ispredFromProbGood and isSURFGood:
#                     bestPrediction[i] = SURFPredictions[i]
#                 else:
#                     bestPrediction[i] = predFromProb[i]
#
#             bestMis = utils.checkMispredictions(test_labels, bestPrediction)
#             print(x, y, utils.countMis(bestMis))
#             if utils.countMis(bestMis) < minimum:
#                 minimum = utils.countMis(bestMis)
#                 minx = x
#                 miny = y
#     print(minx, miny, "Minimum: ", minimum)
#     with open(constants.bestXY, 'wb+') as f:
#         pickle.dump([minx, miny], f)
