# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:06:49 2018

@author: Abhishek Bansal
"""
import csv


def readLabels(filename = "../Functions/Train/Labels.csv"):
    labels = []
    with open(filename) as fp:
        csvfp = csv.reader(fp, delimiter=",")
        for row in csvfp:
            labels.append(row[1])
    return labels

#labels = readLabels()
#print(labels)