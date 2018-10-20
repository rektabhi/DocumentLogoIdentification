# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:41:52 2018

@author: Abhishek Bansal
"""

import numpy as np
from sklearn.externals import joblib
import pickle
from src import chooseBestPrediction, loadImages, SIFT, constants, HOG, SURF, utils


class Model:
    def __init__(self):
        self.SIFTModel = None
        self.SIFTLabels = None
        self.SURFModel = None
        self.SURFLabels = None
        self.HOGModel = None
        self.scaler = None
        self.numOfLogosPerClass = None
        self.x = None
        self.y = None
        self.loadModels()

    def loadModels(self):
        self.loadHOGModel()
        self.loadSIFTModel()
        self.loadSURFModels()
        self.numOfLogosPerClass = utils.numOfLogosPerClass(self.SIFTLabels, constants.numLabels)
        with open(constants.bestXY, 'rb+') as f:
            self.x, self.y = pickle.load(f)

    def loadHOGModel(self):
        self.HOGModel = joblib.load(constants.HOGModelLoc)
        self.scaler = joblib.load(constants.ScalerLoc)

    def loadSIFTModel(self):
        self.SIFTModel = np.load(constants.SIFTModelLoc)
        self.SIFTLabels = np.load(constants.SIFTLabelLoc)

    def loadSURFModels(self):
        self.SURFModel = np.load(constants.SURFModelLoc)
        self.SURFLabels = np.load(constants.SURFLabelLoc)

    def predictLabel(self, image):
        image = loadImages.preprocessImage(image)
        keypoints, descriptors = SIFT.extractSIFTFeatures(image)
        predictedSIFTClass, predictedSIFTConfidence = SIFT.predictSIFTFeatures(
            self.SIFTModel,
            descriptors,
            self.SIFTLabels,
            self.numOfLogosPerClass)
        keypoints, descriptors = SURF.extractSURFFeatures(image)
        predictedSURFClass, predictedSURFConfidence = SURF.predictSURFFeatures(
            self.SURFModel,
            descriptors,
            self.SURFLabels,
            self.numOfLogosPerClass)
        HOGFeatures = HOG.extractHOGFeatures(image)
        HOGFeatures = HOGFeatures.reshape(1, -1)
        HOGFeatures = self.scaler.transform(HOGFeatures)
        predictedHOGClass = self.HOGModel.predict(HOGFeatures)
        HOGProb = self.HOGModel.predict_proba(HOGFeatures)
        predictedValues = dict()
        predictedValues["predictedSIFTClass"] = predictedSIFTClass
        predictedValues["predictedSIFTConfidence"] = predictedSIFTConfidence
        predictedValues["predictedSURFClass"] = predictedSURFClass
        predictedValues["predictedSURFConfidence"] = predictedSURFConfidence
        predictedValues["predictedHOGClass"] = predictedHOGClass
        predictedValues["HOGProb"] = HOGProb
        return chooseBestPrediction.chooseBestPrediction(self, predictedValues)

    def chooseBestPrediction(self):
        return 0

# model = Model()
# model.predictLabel(testImages[0])
