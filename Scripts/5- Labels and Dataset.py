import cv2 as cv
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

vertical = cv.imread("images/Horizontal.png", cv.IMREAD_GRAYSCALE)
vertical = vertical/255
verticalF = vertical.flatten()

horizontal = cv.imread("images/Vertical.png", cv.IMREAD_GRAYSCALE)
horizontal = horizontal/255
horizontalF = horizontal.flatten()

dataSet = np.array([verticalF, horizontalF])
labels = np.array([0,1])

weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

for i in range(10):
    print("Tur : ", i + 1)

    temporaryWeights = weights.copy()
    for image, label in zip(dataSet, labels):
        print("LABEL :: ",label)

        convolution = sum(image * weights)
        print("Convolution :: ", convolution)

        result = sigmoid(convolution)

        error = result - label
        print("Error Value :: ", error)

        adjustment = error * sigmoid_der(result)
        print("Adjustment Rate :: ", adjustment)

        temporaryWeights -= np.dot(image, adjustment)

    weights = temporaryWeights.copy()
    print("Weights Value :: ", list(weights))
    print("")