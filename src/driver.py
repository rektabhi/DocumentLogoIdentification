# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:43:18 2018

@author: Abhishek Bansal
"""

from flask import Flask, request
from src.customPrediction import Model
import numpy as np
import cv2
import json
import base64
import PIL
import io

app = Flask(__name__)


@app.route('/healthcheck')
def healtcheck():
    return 'Server is running!'


@app.route('/predict_image', methods=['POST'])
def predict_image():
    print(request.headers)
    base64Image = base64.b64decode(request.data)
    pilImage = PIL.Image.open(io.BytesIO(base64Image))
    cv2Image = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)

    imFileLoc = "C:/Users/Abhishek Bansal/Desktop/img.jpg"
    cv2.imwrite(imFileLoc, cv2Image)

    response = {}
    response["status"] = 200;
    response["answer"] = "Unknown Answer"
    if cv2Image is None:
        return json.dumps(response)
    return json.dumps(response)
    model = Model()
    prediction = model.predictLabel(cv2Image)
    return prediction
