# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 02:30:06 2018

@author: Abhishek Bansal
"""

import cv2
import numpy as np
from src import constants, utils


class SIFT:
    def __init__(self):
        self.SIFTFeaturesTrain = None
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict()  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.predictions = None
        self.probability = None

    def extractSIFTFeatures(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def countMatchingSIFTFeatures(self, features1, features2):
        matches = self.flann.knnMatch(features1, features2, k=2)
        # ratio test as per Lowe's paper
        count = 0
        for m, n in matches:
            if m.distance < constants.ratioTestLowe * n.distance:
                count += 1
        return count

    def predictSIFTFeatures(self, descriptorsTrain, descriptorTest, trainLabels, numOfLogosPerClass):
        numTrainingExamples = len(descriptorsTrain)
        numLabels = len(numOfLogosPerClass)
        count = np.zeros((numLabels,))
        for i in range(numTrainingExamples):
            count[trainLabels[i] - 1] += self.countMatchingSIFTFeatures(descriptorsTrain[i], descriptorTest)
        count /= numOfLogosPerClass
        return np.argmax(count) + 1, np.amax(count)

    def matchFeatures(self, trainImages, testImages, trainLabels, testlabels, saveModel=True):
        self.SIFTFeaturesTrain = []
        numOfLogosPerClass = utils.numOfLogosPerClass(trainLabels, constants.numLabels)
        for image in trainImages:
            keypoints, descriptors = self.extractSIFTFeatures(image)
            self.SIFTFeaturesTrain.append(descriptors)
        if saveModel:
            np.save(constants.SIFTModelLoc, self.SIFTFeaturesTrain)
            np.save(constants.SIFTLabelLoc, trainLabels)

        self.predictions = []
        self.probability = []
        for index, image in enumerate(testImages):
            keypoints, descriptorTest = self.extractSIFTFeatures(image)
            x, y = self.predictSIFTFeatures(self.SIFTFeaturesTrain, descriptorTest, trainLabels, numOfLogosPerClass)
            self.predictions.append(x)
            self.probability.append(y)

        return self.predictions, self.probability
