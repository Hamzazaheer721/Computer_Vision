from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import time
import cv2 as cv

DATA = './data/'
# window size
N = 1
DMAX = 8


# calculate means of windows of size n and take average, then normalize
# use means to calculate and save variances for later use
# returns normalized image and variances
def pre_proc(im):
    means = np.zeros(im.shape)
    variances = np.zeros(im.shape)
    for i in range(im.shape[0]):
        window = None
        for j in range(im.shape[1]):
            # get mean of window around point
            if window is None:
                # create window array
                window = np.zeros((2 * N + 1, 2 * N + 1))
                for k in range(-N, N + 1):
                    for l in range(-N, N + 1):
                        if i + k >= 0 and i + k < im.shape[0] and j + l >= 0 and j + l < im.shape[1]:
                            window[k + N, l + N] = im[i + k, j + l]
            else:
                for c in range(2 * N + 1):
                    # shift columns
                    if c + 1 < 2 * N + 1:
                        window[:, c] = window[:, c + 1]
                    else:
                        window[:, c] = np.zeros(2 * N + 1)
                # now add new row
                if j + N < im.shape[1]:
                    for k in range(-N, N + 1):
                        if i + k >= 0 and i + k < im.shape[0]:
                            window[k + N, 2 * N] = im[i + k, j + N]
            count = np.count_nonzero(window)
            means[i, j] = np.sum(window) / count

            # calculate variances
            def f(x):
                return x ** 2 - means[i, j] ** 2

            f = np.vectorize(f)

            variances[i, j] = np.sum(f(window)) / count

    return (im - means, variances)


def match(left, right, matches, vars):
    disparity = np.zeros(left.shape)

    def score_fcn(l, r):
        return np.abs(r - l)

    # match from L(r,0) to L(r,last-DMAX)
    for r in range(left.shape[0]):
        for c in range(left.shape[1] - DMAX):
            if vars[r, c] > 1250:
                # for each pixel in L we calculate similarity between L(r,c) and R(r,c to c+DMAX)
                best = np.inf
                dbest = 0
                mins = np.zeros((4, 2))
                mins[:, 1] = np.inf
                for d in range(DMAX):
                    score = score_fcn(left[r, c], right[r, c + d])
                    if score < best:
                        best = score
                        dbest = d
                        matches[r, c] = np.array([r, c + d])

                    for i in range(4):
                        if score < mins[i][1]:
                            mins[3] = mins[i]
                            mins[i] = np.array([d, score])
                            break
                    mins = mins[mins[:, 1].argsort()]

                dd = np.sum(np.abs(mins[:, 0][1:4] - mins[0, 0]))
                if dd > 8:
                    de = np.sum(mins[:, 1][1:4] - mins[0, 1])
                    if mins[0, 1] == 0:
                        ratio = np.inf
                    else:
                        ratio = de / mins[0, 1]
                    if ratio < 5:
                        disparity[r, c] = 0
                        matches[r, c] = np.array([-1, -1])
                        continue

                disparity[r, c] = dbest
                # we need to check for multiple matches to single R pixel
                for x in range(c):
                    if np.array_equal(matches[r, c], matches[r, x]):
                        old_score = score_fcn(left[r, x], right[matches[r, x][0], matches[r, x][1]])
                        if old_score < best:
                            # keep old match
                            matches[r, c] = np.array([-1, -1])
                            disparity[r, c] = 0
                        else:
                            # keep new match, discard old_score
                            matches[r, x] = np.array([-1, -1])
                            disparity[r, x] = 0

    return (matches, disparity)


def main():
    im0 = np.array(Image.open(DATA + 'cones/im2.png').convert('L'))
    im1 = np.array(Image.open(DATA + 'cones/im6.png').convert('L'))

    start = time.time()
    (im0, im0vars) = pre_proc(im0)
    (im1, im1vars) = pre_proc(im1)
    norm_done = time.time()
    print("Normalization took:", norm_done - start)

    # match im0 to im1
    # make array to hold matches with all entries = -1
    # (since this will be array of coordinates, val of -1 means uninitialized)
    matches = np.negative(np.ones((im0.shape[0], im0.shape[1], 2), dtype=np.int))
    (matches, disparity) = match(im0, im1, matches, im0vars)
    matching_done = time.time()
    print("Matching took:", matching_done - norm_done)

    oim0 = cv.imread(DATA + 'cones/im2.png')
    oim1 = cv.imread(DATA + cones/im6.png'')
    statement1 = f"the window size here is decreased to N = {N} so we are getting more detail but more noise as well "
    statement2 = f"the window size here is increased to N = {N} so we are getting lesser detail but smoother disperity maps"
    statement3 = f"the window size here is increased to N = {N} so we are getting even lesser details"
    array = [oim0, oim1, im0, disparity]
    titles = ["image 1 ", "image 2 ", "GrayScaled"]
    for i in range(len(array)):
        plt.subplot(2, 2, i + 1)
        if i == 0 or i == 1:
            im2 = array[i].copy()
            im2[:, :, 0] = array[i][:, :, 2]
            im2[:, :, 2] = array[i][:, :, 0]
            plt.imshow(im2)
            plt.title(titles[i])
        elif i == 2:
            plt.imshow(array[i], cmap="gray")
            plt.title(titles[i])
        elif i >= 2 and N < 4:
            titles.append(statement1)
            plt.imshow(array[i], interpolation='nearest', origin='upper', cmap='gray')
            plt.title(titles[i])
        elif i >= 2 and 4 < N < 10:
            titles.append(statement2)
            plt.imshow(array[i], interpolation='nearest', origin='upper', cmap='gray')
            plt.title(titles[i])
        elif i >= 2 and N >= 10:
            titles.append(statement3)
            plt.imshow(array[i], interpolation='nearest', origin='upper', cmap='gray')
            plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


main()
