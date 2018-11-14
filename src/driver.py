# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:43:18 2018

@author: Abhishek Bansal
"""

from flask import Flask, request
import sys
sys.path.append('..\..\LogoIdentification')
import numpy as np
import cv2
import json
import base64
from PIL import Image
import io
from src.extraction.extractLogo import ExtractLogo


app = Flask(__name__)


@app.route('/healthcheck')
def healthcheck():
    return 'Server is running!'


@app.route('/predict_image', methods=['POST'])
def predict_image():
    print(request.headers)
    base64Image = base64.b64decode(request.data)
    pilImage = Image.open(io.BytesIO(base64Image))
    cv2Image = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)
    print("Extracting Logo!")
    extract = ExtractLogo()
    from src.extraction.detectPage import detectPage
    import src.utils as utils
    cv2Image = detectPage(cv2Image)
    utils.imshow(cv2Image)
    predictedLogoList = extract.extract(cv2Image)
    # imFileLoc = "C:/Users/Abhishek Bansal/Desktop/img.jpg"
    # cv2.imwrite(imFileLoc, cv2Image)

    response = {}
    response["status"] = 200
    response["answer"] = "Predicted logos are: "
    for logo in predictedLogoList:
        response["answer"] += logo
    print(json.dumps(response))
    return json.dumps(response)
