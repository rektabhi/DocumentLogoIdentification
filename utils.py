# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:46:33 2018

@author: Abhishek Bansal
"""
import cv2
import numpy as np


def imshow(image):
    cv2.imshow('image', image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


def numOfLogosPerClass(labels, n):
    numLabels = np.zeros((n, ))
    for label in labels:
        numLabels[label-1]+=1
    return numLabels