# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 02:30:06 2018

@author: Abhishek Bansal
"""

import cv2
import numpy as np
from src import constants


class SIFT:
    def __init__(self):
        self.SIFTFeaturesTrain = None
        self.trainLabels = None
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict()  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.predictions = None
        self.probability = None
        self.numOfLogosPerClass = None

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

    def predictSIFTFeatures(self, descriptorTest):
        numTrainingExamples = len(self.SIFTFeaturesTrain)
        numLabels = len(self.numOfLogosPerClass)
        count = np.zeros((numLabels,))
        for i in range(numTrainingExamples):
            count[self.trainLabels[i] - 1] += self.countMatchingSIFTFeatures(self.SIFTFeaturesTrain[i], descriptorTest)
        count /= self.numOfLogosPerClass
        return np.argmax(count) + 1, np.amax(count)

    def matchFeatures(self, images, saveModel=True):
        self.SIFTFeaturesTrain = []
        self.trainLabels = images.trainLabels
        self.numOfLogosPerClass = images.numOfLogosPerClass
        print("Extracting SIFT Features")
        for image in images.trainImages:
            keypoints, descriptors = self.extractSIFTFeatures(image)
            self.SIFTFeaturesTrain.append(descriptors)
        print("Done!")
        if saveModel:
            np.save(constants.SIFTModelLoc, self.SIFTFeaturesTrain)
            np.save(constants.SIFTLabelLoc, self.trainLabels)
            np.save(constants.SURFNumOfLogosPerClass, self.numOfLogosPerClass)

        self.predictions = []
        self.probability = []

        print("Predicting Test Images - SIFT")
        for index, image in enumerate(images.testImages):
            keypoints, descriptorTest = self.extractSIFTFeatures(image)
            x, y = self.predictSIFTFeatures(descriptorTest)
            self.predictions.append(x)
            self.probability.append(y)
        print("Done!")

        return self.predictions, self.probability

    def loadSIFTModel(self):
        self.SIFTFeaturesTrain = np.load(constants.SIFTModelLoc)
        self.trainLabels = np.load(constants.SIFTLabelLoc)
        self.numOfLogosPerClass = np.load(constants.SIFTNumOfLogosPerClass)
