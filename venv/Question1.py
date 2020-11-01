import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

imgColored = cv.imread("./data/bleach/lines.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(imgColored, cv.COLOR_BGR2GRAY)

# using canny edge detector
dst = cv.Canny(img, 200, 300, None, 3)
cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

# The Standard Hough Transform
lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

# The Probabilistic Hough Line Transform
linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 150, None, 50, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

# reading image for cirle detection
imgCircles = cv.imread("./data/bleach/balls.jpg", cv.IMREAD_COLOR)
imgColored2 = cv.imread("./data/bleach/balls.jpg", cv.IMREAD_COLOR)
img2 = cv.cvtColor(imgColored2, cv.COLOR_BGR2GRAY)
imgColored3 = cv.imread("./data/bleach/balls.jpg", cv.IMREAD_COLOR)
img3 = cv.cvtColor(imgColored3, cv.COLOR_BGR2GRAY)

# applying Median BLur to reduce noise and avoid false circle detection
img2 = cv.medianBlur(img2, 5)
rows2 = img2.shape[0]
img3 = cv.medianBlur(img3, 5)
rows3 = img3.shape[0]

# applying Hough Circle Transform
def houghTransform(img_0, rows, min, max, imgColored_0):
    circles = cv.HoughCircles(img_0, cv.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=min,
                              maxRadius=max)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(imgColored_0, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(imgColored_0, center, radius, (255, 0, 255), 3)


# calling hough circle transform with seperate parameters
houghTransform(img2, rows2, 1, 130, imgColored2)
houghTransform(img3, rows3, 50, 120, imgColored3)

array = [imgColored, cdst, cdstP, imgCircles, imgColored2, imgColored3]
titles = ["Image: 1", "The Standard Hough Line Transform", "The Probabilistic Hough Line Transform", "Image: 2",
          "The Hough Circle Transform(minR=1, MaxR=50)", "The Hough Circle Transform(minR=50, MaxR=120)"]
for i in range(len(array)):
    plt.subplot(2, 3, i + 1)
    if i == 0 or i >= 3:
        im2 = array[i].copy()
        im2[:, :, 0] = array[i][:, :, 2]
        im2[:, :, 2] = array[i][:, :, 0]
        plt.imshow(im2)
    else:
        plt.imshow(array[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
