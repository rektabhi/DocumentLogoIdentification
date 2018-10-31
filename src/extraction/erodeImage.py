import cv2
import src.constants as constants
import src.utils as utils


def erode(image, debug=True):
    # loc = "C:/Users/Abhishek Bansal/Desktop/Image Processing/Logo Identification/Logo.png"
    # image = cv2.imread(loc)

    kernel = constants.kernel_ones
    image = utils.rgb2gray(image)
    if constants.binarizeOriginal:
        image = utils.imbinarize(image)
    dilate_image = cv2.dilate(image, kernel, iterations=constants.numOfDilation)
    binary_dilate_image = utils.imbinarize(dilate_image)
    if debug:
        utils.imshow(dilate_image)
        utils.imshow(binary_dilate_image)
    return binary_dilate_image
