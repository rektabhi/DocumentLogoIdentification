# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:59:52 2018

@author: Abhishek Bansal
"""
import numpy as np


saltAndPepperThreshold = 2000
numLabels = 15
ratioTestLowe = 0.6
numTestImages = 74
numTrainImages = 247
waitTimeImage = 5000
trainImagesLocation = '../data/Train/'
testImagesLocation = '../data/Test/'
trainLabelsLoc = '../data/Train/Labels.csv'
urlLoc = '../data/Train/url.csv'
SIFTModelLoc = '../models/SIFTModel.npy'
SIFTLabelLoc = '../models/SIFTLabels.npy'
SIFTNumOfLogosPerClass = '../models/SIFTNumOfLogosPerClass.npy'
SURFNumOfLogosPerClass = '../models/SURFNumOfLogosPerClass.npy'
SURFModelLoc = '../models/SURFModel.npy'
SURFLabelLoc = '../models/SURFLabels.npy'
HOGModelLoc = '../models/HOGModel'
ScalerLoc = '../models/Scaler'
bestXY = '../models/bestXY'
ADAPTIVE_THRESHOLD = 1


numOfDilation = 6
binarizeOriginal = False
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
kernel_ones = np.ones((3, 3), dtype=np.uint8)
numPixelInSmallComponent = 20
numPixelInLargeComponent = 1000
nearbyFractionVertical = 0.05
nearbyFractionHorizontal = 0.05
percentageAreaVertical = 0.15
percentageAreaHorizontal = 0.15
maxHeight = 600
maxWidth = 600
highNumOfComponent = 1000
