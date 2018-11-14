# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:41:52 2018

@author: Abhishek Bansal
"""


class Predict:

    def __init__(self, ctx):
        self.ctx = ctx
        self.predictedSIFTClass = None
        self.predictedSIFTConfidence = None
        self.predictedSURFClass = None
        self.predictedSURFConfidence = None
        self.predictedHOGClass = None
        self.HOGProb = None

    def predictLabel(self, image):
        image = self.ctx.images.preprocessImage(image)
        sift = self.ctx.sift
        surf = self.ctx.surf
        hog = self.ctx.hog
        keypoints, descriptors = sift.extractSIFTFeatures(image)
        self.predictedSIFTClass, self.predictedSIFTConfidence = sift.predictSIFTFeatures(descriptors)
        keypoints, descriptors = surf.extractSURFFeatures(image)
        self.predictedSURFClass, self.predictedSURFConfidence = surf.predictSURFFeatures(descriptors)
        # HOGFeatures = hog.extractHOGFeatures(image)
        # HOGFeatures = HOGFeatures.reshape(1, -1)
        # HOGFeatures = hog.scaler.transform(HOGFeatures)
        # self.predictedHOGClass = hog.model.predict(HOGFeatures)
        # self.HOGProb = hog.model.predict_proba(HOGFeatures)
        if self.predictedSIFTClass == self.predictedSURFClass:
            return self.predictedSIFTClass
        elif self.predictedSIFTConfidence > self.ctx.y:
            return self.predictedSIFTClass
        elif self.predictedSURFConfidence > self.ctx.y:
            return self.predictedSURFClass
        return -1

    def chooseBestPrediction(self):
        return 0
