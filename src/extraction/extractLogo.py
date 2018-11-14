import src.utils as utils
import src.extraction.erodeImage as erodeImage
import src.extraction.mergeNearbyComponents as merge
from src.extraction.meanConnComp import ConnectedComponents
from src.extraction.segmentLogo import SegmentLogo
from src.customPrediction import Predict
from src.main import Context


class ExtractLogo:
    def __init__(self):
        self.logos = None
        self.document = None
        self.resizeImage = None
        self.erodedImage = None
        self.debug = False
        self.components = None
        self.mergedComponents = None
        self.removedEdgeImage = None
        self.blurredImage = None

    def extract(self, image, debug=False):
        self.document = image
        self.debug = debug

        self.preprocess()
        self.findConnComp()
        # utils.imshow(self.erodedImage)
        sl = SegmentLogo(self.resizeImage, self.mergedComponents.mergedComponents)
        # sl = SegmentLogo(orig_image, comp.components)
        sl.segmentLogoByMean()
        predictedLogoList = []
        for logo in sl.logos:
            utils.imshow(logo)
            # utils.imshow(process_image(logo))
            ctx = Context()
            ctx.loadModels()
            pred = Predict(ctx)
            predictedClass = pred.predictLabel(logo)
            a = pred.predictedSURFClass
            b = pred.predictedSIFTClass
            print(ctx.stringLabels[a-1], ctx.stringLabels[b-1])
            print("Predicted Class: ", ctx.stringLabels[predictedClass-1])
            if predictedClass is not -1:
                predictedLogoList.append(ctx.stringLabels[predictedClass-1])
        return predictedLogoList

    def preprocess(self):
        # self.blurredImage = cv2.GaussianBlur(self.document, (5, 5), 0)
        self.removedEdgeImage = self.document[10:-10][10:-10]
        self.resizeImage = utils.resize(self.removedEdgeImage)
        if self.debug:
            utils.imshow(self.resizeImage)
        self.erodedImage = erodeImage.erode(self.resizeImage, self.debug)
        utils.imshow(self.erodedImage)
        print("Processed Image for Extraction!")
    
    def findConnComp(self):
        self.components = ConnectedComponents(self.erodedImage, self.debug)
        self.components.findMeanOfConnectedComponents()
        print("Found Mean of connected components!")
        print("Number of connected components: ", self.components.numOfComp)
        self.mergedComponents = merge.MergeComponents(self.components)
        sum = 0
        for component in self.components.components:
            sum += component["count"]
        print("Sum: ", sum)
        # mergedComp.highNumOfComponent()
        self.mergedComponents.mergeNearbyComponents()
        print("Components Merged")
        self.mergedComponents.countComponentsBySize()
        print(self.mergedComponents.numOfComponentBySize)
        print("Components grouped")
        self.mergedComponents.removeSmallComponent()
        print("Small Components Removed")
        for component in self.mergedComponents.mergedComponents:
            print(component)
