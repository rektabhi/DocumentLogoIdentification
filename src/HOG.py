# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:51:07 2018

@author: Abhishek Bansal
"""
from src import constants
from src.trainSVM import trainSVM
from skimage import feature
from sklearn import preprocessing
from sklearn.externals import joblib


class HOG:

    def __init__(self):
        self.HOGFeaturesTrain = None
        self.HOGFeaturesTest = None
        self.predictions = None
        self.probability = None
        self.scaler = None
        self.model = None

    def extractHOGFeatures(self, image):
        H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
        return H

    def matchHOGFeatures(self, images, saveModel=True):
        self.HOGFeaturesTrain = []
        for image in images.trainImages:
            self.HOGFeaturesTrain.append(self.extractHOGFeatures(image))

        self.HOGFeaturesTest = []
        for image in images.testImages:
            self.HOGFeaturesTest.append(self.extractHOGFeatures(image))

        self.scaler = preprocessing.StandardScaler()
        self.HOGFeaturesTrain = self.scaler.fit_transform(self.HOGFeaturesTrain)

        if saveModel:
            joblib.dump(self.scaler, constants.ScalerLoc)

        self.HOGFeaturesTest = self.scaler.transform(self.HOGFeaturesTest)
        self.model = trainSVM(self.HOGFeaturesTrain, images.trainLabels)

        if saveModel:
            joblib.dump(self.model, constants.HOGModelLoc)

        self.predictions = self.model.predict(self.HOGFeaturesTest)
        self.probability = self.model.predict_proba(self.HOGFeaturesTest)
        return self.predictions, self.probability

    def loadHOGModel(self):
        self.model = joblib.load(constants.HOGModelLoc)
        self.scaler = joblib.load(constants.ScalerLoc)
