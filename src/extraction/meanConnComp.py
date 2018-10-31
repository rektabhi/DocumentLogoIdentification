import cv2
import numpy as np
import src.utils as utils
import src.extraction.erodeImage as erodeImage
import src.extraction.mergeNearbyComponents as merge


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
                    # print(self.labels[i][j], i, j)
                    self.components[self.labels[i][j]-1]["sumx"] += i
                    self.components[self.labels[i][j]-1]["sumy"] += j
                    self.components[self.labels[i][j]-1]["count"] += 1

        for component in self.components:
            component["meanx"] = int(component["sumx"]/component["count"])
            component["meany"] = int(component["sumy"]/component["count"])

        if self.debug:
            for component in self.components:
                print(component)


# image = erodeImage.erode(image, debug=False)
# print(type(ret))
# print(np.shape(labels))
# print(np.sum(labels))
# Map component labels to hue val
# label_hue = np.uint8(179*labels/np.max(labels))
# blank_ch = 255*np.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
# # cvt to BGR for display
# labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
# # set bg label to black
# labeled_img[label_hue == 0] = 0
#
# utils.imshow(labeled_img)
#
# loc = "C:/Users/Abhishek Bansal/Desktop/Image Processing/Logo Identification/Logo4processed.png"
# cv2.imwrite(loc, labeled_img)


loc = "C:/Users/Abhishek Bansal/Desktop/Image Processing/Logo Identification/Logo.png"
orig_image = cv2.imread(loc)
image = erodeImage.erode(orig_image, debug=False)
loc = "C:/Users/Abhishek Bansal/Desktop/Image Processing/Logo Identification/Logo42222222.png"
cv2.imwrite(loc, image)
comp = ConnectedComponents(image)
comp.findMeanOfConnectedComponents()
mergedComp = merge.MergeComponents()
mergedComp.mergeNearbyComponents(comp)
for component in mergedComp.mergedComponents:
    print(component)
mergedComp.countComponentsBySize()
print(mergedComp.numOfComponentBySize)
from src.extraction.segmentLogo import SegmentLogo
sl = SegmentLogo(orig_image, mergedComp.mergedComponents)
sl.segmentLogoByMean()
for logo in sl.logos:
    utils.imshow(logo)