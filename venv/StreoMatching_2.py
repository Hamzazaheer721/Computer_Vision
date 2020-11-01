import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 as cv
DATA = './data/'
N = 20
DMAX = 30
def stereo_match(left_img, right_img, kernel, max_offset):
    # Load in both images, assumed to be RGBA 8bit per channel images
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)
    w, h = left_img.size  # assume that both images are same size

    # Depth (or disparity) map
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w

    kernel_half = int(kernel / 2)
    offset_adjust = 255 / max_offset  # this is used to map depth map output to 0-255 range

    for y in range(kernel_half, h - kernel_half):
        print(".", end="", flush=True)  # let the user know that something is happening (slowly!)

        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534

            for offset in range(max_offset):
                ssd = 0
                ssd_temp = 0

                # v and u are the x,y of our local window search, used to ensure a good
                # match- going by the squared differences of two pixels alone is insufficient,
                # we want to go by the squared differences of the neighbouring pixels too
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        # iteratively sum the sum of squared differences value for this block
                        # left[] and right[] are arrays of uint8, so converting them to int saves
                        # potential overflow, and executes a lot faster
                        ssd_temp = int(left[y + v, x + u]) - int(right[y + v, (x + u) - offset])
                        ssd += ssd_temp * ssd_temp

                        # if this value is smaller than the previous ssd at this block
                # then it's theoretically a closer match. Store this value against
                # this block..
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset

            # set depth output for this x,y location to the best match
            depth[y, x] = best_offset * offset_adjust


    # Convert to PIL and save it
    Image.fromarray(depth).save(f'depth{N}.png')

    #plotting
    disparity = cv.imread(f'depth{N}.png')
    oim0 = cv.imread(DATA + 'cones/im2.png')
    oim1 = cv.imread(DATA + 'cones/im6.png')
    statement1 = f"the window size here is decreased to N = {N} so we are getting more detail but more noise as well "
    statement2 = f"the window size here is increased to N = {N} so we are getting lesser detail but smoother disperity maps"
    statement3 = f"the window size here is increased to N = {N} so we are getting even lesser details"
    array = [oim0, oim1, disparity]
    titles = ["image 1 ", "image 2 "]
    for i in range(len(array)):
        plt.subplot(2, 2, i + 1)
        if i == 0 or i == 1:
            im2 = array[i].copy()
            im2[:, :, 0] = array[i][:, :, 2]
            im2[:, :, 2] = array[i][:, :, 0]
            plt.imshow(im2)
            plt.title(titles[i])
        elif i > 1 and N < 4:
            titles.append(statement1)
            plt.imshow(array[i])
            plt.title(titles[i])
        elif i > 1 and 4 < N < 10:
            titles.append(statement2)
            plt.imshow(array[i])
            plt.title(titles[i])
        elif i >= 1 and N >= 10:
            titles.append(statement3)
            plt.imshow(array[i])
            plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

stereo_match(DATA + 'cones/im2.png', DATA + 'cones/im6.png', N, DMAX)  # 6x6 local search kernel, 30 pixel search range