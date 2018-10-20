# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:59:52 2018

@author: Abhishek Bansal
"""

saltAndPepperThreshold = 2000
numLabels = 15
ratioTestLowe = 0.6
numTestImages = 74
waitTimeImage = 5000
trainImagesLocation = '../data/Train/'
testImagesLocation = '../data/Test/'
trainLabelsLoc = '../data/Train/Labels.csv'
SIFTModelLoc = '../models/SIFTModel.npy'
SIFTLabelLoc = '../models/SIFTLabels.npy'
SURFModelLoc = '../models/SURFModel.npy'
SURFLabelLoc = '../models/SURFLabels.npy'
HOGModelLoc = '../models/HOGModel'
ScalerLoc = '../models/Scaler'
bestXY = '../models/bestXY'
