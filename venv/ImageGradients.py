import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

img = cv.imread('./data/bleach/ichigo.jpg', 0)
lap = cv.Laplacian(img, cv.CV_64F, ksize=1)
lap = np.uint8(np.absolute(lap))
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=1)
edges = cv.Canny(img, 100, 200, None, 3)

sobelCombined = cv.bitwise_or(sobelX,sobelY)

imgArray = [img, lap, sobelX, sobelY, sobelCombined, edges]
imgTitles = ["Image", "Laplacian", "SobelX", "SobelY", "Sobel Combined", "Canny"]
for x in range(len(imgArray)):
    plt.subplot(2, 3, x+1)

    # using following 4 lines of code is for avoiding the blue color that matplotlib adds to colored image
    # im2 = imgArray[x].copy()
    # im2[:, :, 0] = imgArray[x][:, :, 2]
    # im2[:, :, 2] = imgArray[x][:, :, 0]
    # plt.imshow(im2)

    # use this following single line if you want to show your converted gray img into grayscale on matplot
    plt.imshow(imgArray[x], "gray")
    plt.title(imgTitles[x])
    plt.xticks([]), plt.yticks([])
plt.show()