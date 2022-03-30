import cv2 as cv
import numpy as np

vertical = cv.imread("images/Horizontal.png", cv.IMREAD_GRAYSCALE)
horizontal = cv.imread("images/Vertical.png", cv.IMREAD_GRAYSCALE)

vertical = vertical/255
horizontal = horizontal/255

verticalF = vertical.flatten()
horizontalF = horizontal.flatten()

def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

dataset = np.array([verticalF, horizontalF])
labels = np.array([0, 1]).T

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        print(self.weights)

    def feed_forward(self, input):
        weighted_sum = np.dot(input, self.weights)
        output = sigmoid(weighted_sum)
        return output

    def back_propagation(self, input, output, labels):
        error = output - labels
        adjustment = error * sigmoid_der(output)
        weights_update = np.dot(input.T, adjustment)
        self.weights -= weights_update

layer1 = Layer(9, 1)

for i in range(100):
    output = layer1.feed_forward(dataset)
    layer1.back_propagation(dataset, output, labels)