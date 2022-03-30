import cv2 as cv
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

label = "VERTICAL"
vertical = cv.imread("images/Horizontal.png", cv.IMREAD_GRAYSCALE)
vertical = vertical/255
verticalF = vertical.flatten()

weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
convolution = sum(verticalF * weights)

result = sigmoid(convolution)

if result < 0.5:
    if label == "VERTICAL":
        print("VERTICAL")
    elif label != result:
        print("Eşleşme hatası (Nesne vertical sınıfında tahmin edildi)")
elif result > 0.5:
    if label == "HORIZONTAL":
        print("HORIZONTAL")
    elif label != result:
        print("Eşleşme hatası (Nesne horizontal sınıfında tahmin edildi)")