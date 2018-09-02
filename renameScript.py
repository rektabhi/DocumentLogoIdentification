# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:16:25 2018

@author: Abhishek Bansal
"""


import os
dire = "../Functions/Test/"
for index in range(1, 75):
    filenamewithpath = os.path.join(dire, str(index))
    # new_file_num = int(file) - 1
#    dire = "../Functions/Test/New/"
    new_file_name = os.path.join(dire, str(index-1))
    print(filenamewithpath, new_file_name)
    os.rename(filenamewithpath, new_file_name)