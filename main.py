# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:16:07 2018

@author: Abhishek Bansal
"""
import numpy as np
import LoadImages
import utils
import constants
from extractHOGFeatures import extractHOGFeatures
from readLabels import readLabels
from trainSVM import trainSVM
from sklearn import svm
from classLabels import y as train_labels, y2 as test_labels
from sklearn import preprocessing
import SURF


#trainImages = LoadImages.loadTrainImages()
#testImages = LoadImages.loadTestImages()
#stringLabels = readLabels()

def HOG():
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
    #print(model)
    #print(np.max(HOGFeaturesTrain))
    #for image_feature in HOGFeaturesTrain:
    predictions = model.predict(HOGFeaturesTrain)
    #print(predictions != train_labels)
    predictions = []
    predictions = model.predict(HOGFeaturesTest)
    print(predictions)
    print(test_labels)
    count = 0
    print(predictions != test_labels)
    for index in range(np.size(test_labels)):
        if predictions[index] != test_labels[index]:
            print(index)
            print(stringLabels[predictions[index]-1], stringLabels[test_labels[index]-1])
            utils.imshow(testImages[index])
            count+=1
    print(count)
    
SURF.matchFeatures(trainImages, testImages, train_labels, test_labels)
