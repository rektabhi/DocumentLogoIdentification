# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:07:43 2018

@author: Abhishek Bansal
"""

import cv2
import numpy as np
from src import constants


class SURF:
    def __init__(self):
        self.SURFFeaturesTrain = None
        self.trainLabels = None
        self.surf = cv2.xfeatures2d.SURF_create()
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict()  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.predictions = None
        self.probability = None
        self.numOfLogosPerClass = None

    def extractSURFFeatures(self, image):
        keypoints, descriptors = self.surf.detectAndCompute(image, None)
        return keypoints, descriptors

    def countMatchingSURFFeatures(self, features1, features2):
        matches = self.flann.knnMatch(features1, features2, k=2)
        # ratio test as per Lowe's paper
        count = 0
        for m, n in matches:
            if m.distance < constants.ratioTestLowe * n.distance:
                count += 1
        return count

    def predictSURFFeatures(self, descriptorTest):
        numTrainingExamples = len(self.SURFFeaturesTrain)
        numLabels = len(self.numOfLogosPerClass)
        count = np.zeros((numLabels,))
        for i in range(numTrainingExamples):
            count[self.trainLabels[i] - 1] += self.countMatchingSURFFeatures(self.SURFFeaturesTrain[i], descriptorTest)
        count /= self.numOfLogosPerClass
        return np.argmax(count) + 1, np.amax(count)

    def matchFeatures(self, images, saveModel=True):
        self.SURFFeaturesTrain = []
        self.trainLabels = images.trainLabels
        self.numOfLogosPerClass = images.numOfLogosPerClass
        print("Extracting SURF Features")
        for image in images.trainImages:
            keypoints, descriptors = self.extractSURFFeatures(image)
            self.SURFFeaturesTrain.append(descriptors)
        print("Done!")
        if saveModel:
            np.save(constants.SURFModelLoc, self.SURFFeaturesTrain)
            np.save(constants.SURFLabelLoc, self.trainLabels)

        self.predictions = []
        self.probability = []

        print("Predicting Test Images - SURF")
        for index, image in enumerate(images.testImages):
            keypoints, descriptorTest = self.extractSURFFeatures(image)
            x, y = self.predictSURFFeatures(descriptorTest)
            self.predictions.append(x)
            self.probability.append(y)
        print("Done!")

        return self.predictions, self.probability

    def loadSURFModel(self):
        self.SURFFeaturesTrain = np.load(constants.SURFModelLoc)
        self.trainLabels = np.load(constants.SURFLabelLoc)
        # TODO: Add numOfLogosPerClass Initialization here


