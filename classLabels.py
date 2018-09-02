# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:28:25 2018

@author: Abhishek Bansal
"""
import numpy as np


y = np.zeros((147,), dtype=np.uint8)
y2 = np.zeros((74,), dtype=np.uint8)

# Training Data
y[0:11]=1
y[11:18]=2
y[18:25]=3
y[25:32]=4
y[32:40]=5
y[40:50]=6
y[50:58]=7
y[58:63]=8
y[63:68]=9
y[68:74]=10
y[74:88]=11
y[88:102]=12
y[102:117]=13
y[117:133]=14
y[133:147]=15

# Test Data
y2[0:3]=2
y2[3:9]=3
y2[9:12]=4
y2[12:19]=1
y2[19:34]=5
y2[34:39]=6
y2[39:44]=7
y2[44:48]=10
y2[48:50]=9
y2[50:53]=8
y2[53:57]=11
y2[57:62]=12
y2[62:66]=13
y2[66:70]=14
y2[70:74]=15

#print(y)
#print(y2)