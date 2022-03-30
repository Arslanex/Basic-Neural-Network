import cv2 as cv
import numpy as np

def prepare_the_image(imgPath):
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    img = img/255
    flatten = img.flatten()
    return flatten

vertical = prepare_the_image("images/Horizontal.png")
horizontal = prepare_the_image("images/Vertical.png")

count = 0
while True:
    count += 1
    filter = np.random.randint(-1, 2, size=9)
    convolution_vertical = sum(vertical * filter)
    convolution_horizontal = sum(horizontal * filter)
    if convolution_horizontal != convolution_vertical:
        print("Uygun Eşleşme Bulundu")
        print("Filtre: ", filter)
        print(convolution_vertical)
        print(convolution_horizontal)
        print("Deneme Sayısı: ", count)
        break