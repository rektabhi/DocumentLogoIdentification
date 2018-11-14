import numpy as np
import src.constants as constants


class MergeComponents:
    
    def __init__(self, connectedComponents):
        self.parentArray = None
        self.thresholdHorizontal = None
        self.thresholdVertical = None
        self.cc = connectedComponents
        self.mergedComponents = None
        self.maxPixelCountInComponent = None
        self.numOfComponentBySize = {"small": 0, "medium": 0, "large": 0}

    def mergeNearbyComponents(self):
        self.parentArray = np.zeros((self.cc.numOfComp - 1,), dtype=np.int8)
        for i in range(self.cc.numOfComp - 1):
            self.parentArray[i] = i
        self.thresholdHorizontal = int(np.shape(self.cc.image)[0] * constants.nearbyFractionHorizontal)
        self.thresholdVertical = int(np.shape(self.cc.image)[1] * constants.nearbyFractionVertical)
        i = 0
        while i < self.cc.numOfComp-1:
            j = i + 1
            while j < self.cc.numOfComp-1:
                if((np.abs(self.cc.components[i]["meanx"] - self.cc.components[j]["meanx"]) < self.thresholdHorizontal)
                and (np.abs(self.cc.components[i]["meany"] - self.cc.components[j]["meany"]) < self.thresholdVertical)):
                    if self.getParent(self.parentArray, i) != self.getParent(self.parentArray, j):
                        x = self.getParent(self.parentArray, i)
                        y = self.getParent(self.parentArray, j)
                        self.parentArray[x] = y
                j += 1
            i += 1
    
        parentMerge = np.zeros((self.cc.numOfComp - 1,), dtype=np.int8)
    
        for i in range(self.cc.numOfComp - 1):
            parentMerge[i] = self.getParent(self.parentArray, i)
    
        k = -1
        self.mergedComponents = []
        for i in range(self.cc.numOfComp - 1):
            if parentMerge[i] != -1:
                k += 1
                self.mergedComponents.append(self.cc.components[i])
                for j in range(i+1, self.cc.numOfComp - 1):
                    if parentMerge[j] != -1 and self.getParent(self.parentArray, i) == self.getParent(self.parentArray, j):
                        self.mergedComponents[k]["sumx"] += self.cc.components[j]["sumx"]
                        self.mergedComponents[k]["sumy"] += self.cc.components[j]["sumy"]
                        self.mergedComponents[k]["count"] += self.cc.components[j]["count"]
                        self.mergedComponents[k]["meanx"] = int(self.mergedComponents[k]["sumx"]/self.mergedComponents[k]["count"])
                        self.mergedComponents[k]["meany"] = int(self.mergedComponents[k]["sumy"]/self.mergedComponents[k]["count"])
                        parentMerge[j] = -1
                parentMerge[i] = -1
        return self.mergedComponents
    
    def getParent(self, parentArray, i):
        while parentArray[i] != i:
            i = parentArray[i]
        return i

    def removeSmallComponent(self):
        if self.numOfComponentBySize["large"] > 0:
            self.removeSmallComponentFromList()
        elif self.numOfComponentBySize["medium"] > 0:
            self.removeSmallComponentFromList()
        elif self.numOfComponentBySize["small"] > 2:
            self.mergedComponents = sorted(self.mergedComponents, key=lambda k: k["count"])
            self.mergedComponents = self.mergedComponents[-2:-1]

    def removeSmallComponentFromList(self):
        for index, component in enumerate(self.mergedComponents):
            if component["count"] < constants.numPixelInSmallComponent:
                self.mergedComponents[index] = None
        self.mergedComponents = [component for component in self.mergedComponents if component is not None]

    def countComponentsBySize(self, componentList=None):
        if componentList is None:
            componentList = self.mergedComponents
        self.maxPixelCountInComponent = 0
        for component in componentList:
            if component["count"]>self.maxPixelCountInComponent:
                self.maxPixelCountInComponent = component["count"]
            if component["count"] > constants.numPixelInLargeComponent:
                self.numOfComponentBySize["large"] += 1
            elif component["count"] < constants.numPixelInSmallComponent:
                self.numOfComponentBySize["small"] += 1
            else:
                self.numOfComponentBySize["medium"] += 1

    def highNumOfComponent(self):
        if self.cc.numOfComp > constants.highNumOfComponent:
            self.removeSmallComponentFromList()
