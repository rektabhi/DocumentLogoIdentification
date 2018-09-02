# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 21:07:43 2018

@author: Abhishek Bansal
"""

import cv2
import utils
import numpy as np
import constants


def extractSURFFeatures(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return keypoints, descriptors


def countMatchingSURFFeatures(features1, features2):
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


def predictSURFFeatures(descriptorsTrain, descriptorTest, trainLabels, numOfLogosPerClass):
    numTrainingExamples = len(descriptorsTrain)
    numLabels = len(numOfLogosPerClass)
    count = np.zeros((numLabels, ))
    for i in range(numTrainingExamples):
        count[trainLabels[i]-1] += countMatchingSURFFeatures(descriptorsTrain[i], descriptorTest)
    count/=numOfLogosPerClass
    return (np.argmax(count))

    
def matchFeatures(trainImages, testImages, trainLabels, testlabels):
    SURFFeaturesTrain = []
    numOfLogosPerClass = utils.numOfLogosPerClass(trainLabels, constants.numLabels)
    for image in trainImages:
        keypoints, descriptors = extractSURFFeatures(image)
        SURFFeaturesTrain.append(descriptors)
        
    count = 0
    for index, image in enumerate(testImages):
        keypoints, descriptors = extractSURFFeatures(image)
        prediction = predictSURFFeatures(SURFFeaturesTrain, descriptors, trainLabels, numOfLogosPerClass)
        if prediction != testlabels[index]-1:
#            utils.imshow(image)
            print(prediction, testlabels[index])
            count+=1
    
    print(count)
        