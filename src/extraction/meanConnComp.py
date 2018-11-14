import cv2
import numpy as np
import src.utils as utils


class ConnectedComponents:

    def __init__(self, image, debug=True):
        self.image = image
        self.numOfComp = None
        self.labels = None
        self.components = None
        self.debug = debug

    def findConnectedComponents(self):
        gray_image = utils.rgb2gray(self.image)
        binary_image = utils.imbinarize(gray_image)
        complement_image = utils.imcomplement(binary_image)
        self.numOfComp, self.labels = cv2.connectedComponents(complement_image)

    def findMeanOfConnectedComponents(self):
        self.findConnectedComponents()

        keys = ["sumx", "sumy", "meanx", "meany", "count"]
        component = dict.fromkeys(keys, 0)
        # Initialize the list of dictionaries
        self.components = [component.copy() for _ in range(self.numOfComp-1)]

        for i in range(np.shape(self.labels)[0]):
            for j in range(np.shape(self.labels)[1]):
                if self.labels[i][j] != 0:
                    self.components[self.labels[i][j]-1]["sumx"] += i
                    self.components[self.labels[i][j]-1]["sumy"] += j
                    self.components[self.labels[i][j]-1]["count"] += 1

        for component in self.components:
            component["meanx"] = int(component["sumx"]/component["count"])
            component["meany"] = int(component["sumy"]/component["count"])

        if self.debug:
            for component in self.components:
                print(component)
