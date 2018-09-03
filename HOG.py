# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:51:07 2018

@author: Abhishek Bansal
"""
import cv2
import numpy as np
import utils
import constants
from skimage import feature
from trainSVM import trainSVM
from sklearn import svm
from classLabels import y as train_labels, y2 as test_labels
from sklearn import preprocessing


def extractHOGFeatures(image):
    H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), 
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return H


def matchHOGFeatures(trainImages, testImages, trainLabels, testLabels):
    HOGFeaturesTrain = []
    for image in trainImages:
        HOGFeaturesTrain.append(extractHOGFeatures(image))
    
    HOGFeaturesTest = []
    for image in testImages:
        HOGFeaturesTest.append(extractHOGFeatures(image))
    
    scaler = preprocessing.StandardScaler()
    HOGFeaturesTrain = scaler.fit_transform(HOGFeaturesTrain)
    HOGFeaturesTest = scaler.transform(HOGFeaturesTest)
    model = trainSVM(HOGFeaturesTrain, train_labels)
    predictions = model.predict(HOGFeaturesTest)
    prob = model.predict_proba(HOGFeaturesTest)
    return predictions, prob
