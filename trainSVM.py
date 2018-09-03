# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:26:06 2018

@author: Abhishek Bansal
"""

from sklearn import svm

def trainSVM(X, y):
    model = svm.SVC(probability=True)
    model.fit(X, y)
    return model