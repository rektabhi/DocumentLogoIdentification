import requests
from customPrediction import Model
import cv2
import numpy as np

addr = 'http://192.168.137.129:5000'
test_url = addr + '/predict_image'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


def post_image():
    image_file = '../Functions/Train/100.png'
    image = open(image_file, 'rb').read()
    response = requests.post(test_url, data=image, headers=headers)
    print(response.content)
post_image()

def check():
    model = Model()
    image_file = '../Functions/Train/100.png'
    with open(image_file, 'rb') as f:    
        image = f.read()
    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(model.predictLabel(image))

#check()



#image_file = '../Scan0019.jpg'
#f = open(image_file, 'rb')
#image = f.read()
#f.close()
#nparr = np.fromstring(image, np.uint8)
#image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)





