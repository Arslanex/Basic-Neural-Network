import cv2 as cv
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feed_forward(input, weights):
    convolution = sum(input * weights)
    result = sigmoid(convolution)
    return result

def back_propagation(result, label, input):
    error = result - label
    adjustment = error * sigmoid_der(result)
    weightUpdate = np.dot(input, adjustment)
    return weightUpdate

vertical = cv.imread("images/Horizontal.png", cv.IMREAD_GRAYSCALE)
horizontal = cv.imread("images/Vertical.png", cv.IMREAD_GRAYSCALE)

vertical = vertical/255
horizontal = horizontal/255

verticalF = vertical.flatten()
horizontalF = horizontal.flatten()

dataSet = np.array([verticalF, horizontalF])
labels = np.array([0,1])

weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

for i in range(10):
    print("Round: ", i + 1)

    result = feed_forward(dataSet, weights)
    weights -= back_propagation(result, labels, dataSet)

print("Weights", weights)
