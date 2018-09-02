# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 00:02:18 2018

@author: Abhishek Bansal
"""
import cv2
import numpy as np
import utils


def detectNoise(image):
    m, n = np.shape(image)
    padarr = np.zeros((m+2, n+2), dtype=np.uint8)
    for i in range(1,m+1):
        for j in range(1,n+1):
            padarr[i][j] = image[i-1][j-1]
    count = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            window = padarr[i-1:i+2, j-1:j+2]
            window_sum = 0
            for ii in range(3):
                for jj in range(3):
                    window_sum += window[ii][jj]
            if window_sum == 255 or window_sum == 255*8:
                count+=1
    return count

#print(detectNoise(testImages[20]))