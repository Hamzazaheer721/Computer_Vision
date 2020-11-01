import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def ratioTest(matches):
    goodMatches = []
    for a, b in matches:
        if a.distance < 0.75 * b.distance:
            goodMatches.append([a])
    return goodMatches


img1 = cv.imread('D:/images/im2.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('D:/images/im6.png', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('D:/images/disp2.png', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('D:/images/disp6.png', cv.IMREAD_GRAYSCALE)
img5 = cv.imread('D:/images/in5.png', cv.IMREAD_GRAYSCALE)
img6 = cv.imread('D:/images/in6.png', cv.IMREAD_GRAYSCALE)
img7 = cv.imread('D:/images/in7.png', cv.IMREAD_GRAYSCALE)
img8 = cv.imread('D:/images/in8.png', cv.IMREAD_GRAYSCALE)

array = [img2, img3, img4, img5, img6, img7, img8]

# we will compare the first image img1 with rest of the images in array
sift = cv.xfeatures2d.SIFT_create();
kp1, dp1 = sift.detectAndCompute(img1, None)

# Brute Force Matching
bf = cv.BFMatcher()

# Iterating Array
highestSimilarity = 0
finalSentence = ""
for i in range(len(array)):
    img = array[i];
    kpY, dpY = sift.detectAndCompute(img, None)
    matches = bf.knnMatch(dp1, dpY, k=2)

    # Applying ratio test
    gm = ratioTest(matches)
    goodMatches1 = len(gm)
    print(f"After Ratio Test:  the Similarity Measure between Image: 1 and Image: {i + 2} is {goodMatches1}. \n")

    if (goodMatches1 > highestSimilarity):
        highestSimilarity = goodMatches1
        finalSentence = f"After Iteration: The highest Similarity Measure is : {highestSimilarity} " + f"between between Image: 1 and Image:{i + 2}"
        index = i;
        kp2 = kpY;
print(finalSentence)
imageToCompare = array[index];
img3 = cv.drawMatchesKnn(img1, kp1, imageToCompare, kp2, gm, None, flags=2)


plt.imshow(img3)
plt.title(finalSentence)
plt.show()