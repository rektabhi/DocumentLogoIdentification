# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:16:07 2018

@author: Abhishek Bansal
"""
import numpy as np
import LoadImages
import utils
import constants

from readLabels import readLabels
from classLabels import y as train_labels, y2 as test_labels
import SURF
import HOG


trainImages = LoadImages.loadTrainImages()
testImages = LoadImages.loadTestImages()
stringLabels = readLabels()

HOG.HOG(trainImages, testImages, train_labels, test_labels)
SURF.matchFeatures(trainImages, testImages, train_labels, test_labels)
