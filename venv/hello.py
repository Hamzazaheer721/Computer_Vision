import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/Ricardo-PC/im2.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
surf = cv.xfeatures2d.SURF_create()
orb = cv.ORB_create()
kp = sift.detect(gray, None)

keypoints, discriptors = orb.detectAndCompute(img, None)
# replace orb with sift or surf to know about key points and desciptor of respective ones
'''img = cv.drawKeypoints(gray, kp, img) 
cv.imwrite('sift_keypoints.jpg', img) '''

img = cv.drawKeypoints(gray, keypoints, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# replace kp to simply know key points

cv.imwrite('sift_keypoints.jpg', img)

cv.imshow("im2", img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)