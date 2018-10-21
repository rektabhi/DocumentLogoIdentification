# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:28:25 2018

@author: Abhishek Bansal
"""
import numpy as np

trainLabels = np.zeros((147,), dtype=np.uint8)
testLabels = np.zeros((74,), dtype=np.uint8)

# Training Data
trainLabels[0:11] = 1
trainLabels[11:18] = 2
trainLabels[18:25] = 3
trainLabels[25:32] = 4
trainLabels[32:40] = 5
trainLabels[40:50] = 6
trainLabels[50:58] = 7
trainLabels[58:63] = 8
trainLabels[63:68] = 9
trainLabels[68:74] = 10
trainLabels[74:88] = 11
trainLabels[88:102] = 12
trainLabels[102:117] = 13
trainLabels[117:133] = 14
trainLabels[133:147] = 15

# Test Data
testLabels[0:3] = 2
testLabels[3:9] = 3
testLabels[9:12] = 4
testLabels[12:19] = 1
testLabels[19:34] = 5
testLabels[34:39] = 6
testLabels[39:44] = 7
testLabels[44:48] = 10
testLabels[48:50] = 9
testLabels[50:53] = 8
testLabels[53:57] = 11
testLabels[57:62] = 12
testLabels[62:66] = 13
testLabels[66:70] = 14
testLabels[70:74] = 15
