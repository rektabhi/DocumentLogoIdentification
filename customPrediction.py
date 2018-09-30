# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:41:52 2018

@author: Abhishek Bansal
"""

import numpy as np
import SURF
import SIFT
import cv2
import constants
import utils
import LoadImages
import HOG
from sklearn.externals import joblib
import pickle

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
        image = LoadImages.preprocessImage(image)
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
        print(predictedHOGClass)
        return 'Prediction!'
        
    def chooseBestPrediction(self):
        return 0

#model = Model()
#model.predictLabel(testImages[0])



