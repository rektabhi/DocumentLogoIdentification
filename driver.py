# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:43:18 2018

@author: Abhishek Bansal
"""

from flask import Flask, request
import numpy as np
import cv2
app = Flask(__name__)
from customPrediction import Model

@app.route('/healtcheck')
def healtcheck():
    return 'Server is running!'


@app.route('/predict_image', methods=['POST'])
def predict_image():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    model = Model()
    prediction = model.predictLabel(image)
    return prediction



