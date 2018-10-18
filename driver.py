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
import json
import base64
import PIL
import io


@app.route('/healthcheck')
def healtcheck():
    return 'Server is running!'


@app.route('/predict_image', methods=['POST'])
def predict_image():
    print(request.headers)
    base64image = base64.b64decode(request.data)
    pil_image = PIL.Image.open(io.BytesIO(base64image))
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    imFileLoc = "C:/Users/Abhishek Bansal/Desktop/img.jpg"
    cv2.imwrite(imFileLoc, opencvImage)




    response = {}
    response["status"] = 200;
    response["answer"] = "Unknown Answer"
    if opencvImage is None:
        return json.dumps(response)
    return json.dumps(response)
    model = Model()
    prediction = model.predictLabel(opencvImage)
    return prediction



