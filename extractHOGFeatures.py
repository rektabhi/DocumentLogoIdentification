# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:55:34 2018

@author: Abhishek Bansal
"""

from skimage import feature
def extractHOGFeatures(image):
    H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8), 
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
    return H