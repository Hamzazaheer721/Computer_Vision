import numpy
import cv2 as cv

img1 = cv.imread('D:/cones/im2.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('D:/cones/im6.png', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('D:/cones/disp2.png', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('D:/cones/disp6.png', cv.IMREAD_GRAYSCALE)

array = [img2, img3, img4]
similaritiesArray = []

# we will compare the first image img1 with rest of the images in array

orb = cv.ORB_create();
sift = cv.xfeatures2d.SIFT_create();
kp1, dp1 = orb.detectAndCompute(img1, None)

# Brute Force Matching

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
kpX, dpX = orb.detectAndCompute(img2, None)
matches = bf.match(dp1, dpX)
print(len(matches))

# for x in array:
#     kpX, dpX = sift.detectAndCompute(array[x], None)
#     matches = bf.match(dp1, dpX)
#     print (matches)







# for d in img2:
#     print(d)
# cv.imshow("image1", img1)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.waitKey(1)
