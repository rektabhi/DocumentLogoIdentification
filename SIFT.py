# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 02:30:06 2018

@author: Abhishek Bansal
"""

import cv2
import utils
import numpy as np
import constants


def extractSIFTFeatures(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def countMatchingSIFTFeatures(features1, features2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict()   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(features1, features2,k=2)
    # ratio test as per Lowe's paper
    count = 0
    for m,n in matches:
        if m.distance < constants.ratioTestLowe*n.distance:
            count+=1
    return count


def predictSIFTFeatures(descriptorsTrain, descriptorTest, trainLabels, numOfLogosPerClass):
    numTrainingExamples = len(descriptorsTrain)
    numLabels = len(numOfLogosPerClass)
    count = np.zeros((numLabels, ))
    for i in range(numTrainingExamples):
        count[trainLabels[i]-1] += countMatchingSIFTFeatures(descriptorsTrain[i], descriptorTest)
    count/=numOfLogosPerClass
    return np.argmax(count)+1, np.amax(count)

    
def matchFeatures(trainImages, testImages, trainLabels, testlabels, saveModel=True):
    SIFTFeaturesTrain = []
    numOfLogosPerClass = utils.numOfLogosPerClass(trainLabels, constants.numLabels)
    for image in trainImages:
        keypoints, descriptors = extractSIFTFeatures(image)
        SIFTFeaturesTrain.append(descriptors)
    if saveModel:
        np.save(constants.SIFTModelLoc, SIFTFeaturesTrain)
        np.save(constants.SIFTLabelLoc, trainLabels)
        
    predictions = []
    predProb = []
    for index, image in enumerate(testImages):
        keypoints, descriptors = extractSIFTFeatures(image)
        x, y = predictSIFTFeatures(SIFTFeaturesTrain, descriptors, trainLabels, numOfLogosPerClass)
        predictions.append(x)
        predProb.append(y)

    return predictions, predProb
    