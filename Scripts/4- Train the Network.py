import cv2 as cv
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

label = "VERTICAL"
vertical = cv.imread("images/Horizontal.png", cv.IMREAD_GRAYSCALE)
vertical = vertical/255
verticalF = vertical.flatten()

weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

for i in range(10):
    print("Tur : ", i + 1)

    convolution = sum(verticalF * weights)
    result = sigmoid(convolution)

    if result < 0.5:
        print("Tespit Soncu :: Vertical")
    elif result > 0.5:
        print("Tespit Sonucu :: Horizontal")

    error = result - 0
    print("Error Value :: ", error)

    adjustment = error * sigmoid_der(result)
    print("Adjustment Rate :: ", adjustment)

    weights -= np.dot(verticalF, adjustment)
    print("Ağırlık Değerleri :: ", list(weights))
    print()