import requests
from customPrediction import Model
import cv2

addr = 'http://192.168.137.129:5000'
test_url = addr + '/predict_image'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


def post_image():
    img_file = '../Scan0019.jpg'
    img = open(img_file, 'rb').read()
    response = requests.post(test_url, data=img, headers=headers)
    print(response.content)
#post_image()

def check():
    model = Model()
    image_file = '../Scan0019.jpg'
    image = open(image_file, 'rb').read()
#    print(model.predictLabel(image))
check()

image_file = '../Scan0019.jpg'
f = open(image_file, 'rb')
image = f.read()
print(np.size(image))
f.close()




