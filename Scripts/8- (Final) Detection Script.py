import cv2 as cv
import numpy as np

def prepare_the_image(imgPath):
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (3,3), interpolation=0)
    img = img/255
    flatten = img.flatten()
    return flatten

imgF = prepare_the_image("images/asymmetric_vertical.png")

weights = [0.70167027, 2.19933889, 0.15183252, -1.52314988, -0.61637916, -1.72133963, 0.97506747, 2.05574302, 0.86215589]
convolution = sum(imgF * weights)

if convolution < 0.5:
    print("VERTICAL")
elif convolution > 0.5:
    print("HORIZONTAL")
