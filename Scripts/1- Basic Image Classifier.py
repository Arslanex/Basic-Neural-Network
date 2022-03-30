import cv2 as cv

def vertical_or_horizontal(img):
    img = img/255
    imgF = img.flatten()
    filter = [1, -1, 1, -1, 1, -1, 1, 2, 1]
    result = sum(imgF * filter)

    if result == -1.0:
        print("Vertical")
    else:
        print("Horizontal")
    cv.waitKey(0)

img = cv.imread("images/Vertical.png", cv.IMREAD_GRAYSCALE)
vertical_or_horizontal(img)