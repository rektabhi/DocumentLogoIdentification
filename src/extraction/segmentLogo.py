import numpy as np
import src.constants as constants


class SegmentLogo:

    def __init__(self, document, components):

        self.logos = []
        self.document = document
        self.components = components

    def segmentLogoByMean(self):

        height = np.shape(self.document)[1]
        width = np.shape(self.document)[0]
        print(height, width)
        for component in self.components:
            top = component["meanx"] - constants.percentageAreaVertical*width
            bottom = component["meanx"] + constants.percentageAreaVertical*width
            left = component["meany"] - constants.percentageAreaHorizontal*height
            right = component["meany"] + constants.percentageAreaHorizontal*height
            print(left, right, top, bottom)
            if top < 0:
                top = 0
            if bottom < 0:
                bottom = 0
            if left < 0:
                left = 0
            if right < 0:
                right = 0
            if top > width:
                top = width
            if bottom > width:
                bottom = width
            if left > height:
                left = height
            if right > height:
                left = height
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            logo = self.document[top:bottom-1, left:right-1]
            self.logos.append(logo)
            print(left, right, top, bottom)
