import matplotlib.pyplot as plt
import numpy as np
import cv2
import webcolors


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS2_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


# read images
river_mod = cv2.imread('Images/River_Mod.jpg')
river_mac = cv2.imread('Images/River_Mac.jpg')
boston_mod = cv2.imread('Images/Boston_Mod.jpg')
boston_mac = cv2.imread('Images/Boston_Mac.jpg')
rural_mod = cv2.imread('Images/Rural_Mod.jpg')
rural_mac = cv2.imread('Images/Rural_Mac.jpg')

gray1 = cv2.cvtColor(boston_mod, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(boston_mac, cv2.COLOR_BGR2GRAY)

colors = {}
for row in boston_mac:
    for pixel in row:
        color = closest_colour(pixel)
        if color in colors.keys():
            colors[color] += 1
        else:
            colors[color] = 1
print(colors)

# rgb_weights = [0.2989, 0.5870, 0.1140]
#
# grayscale_image = np.dot(img1[..., :3], rgb_weights)
# plt.imshow(img1)
# plt.show()
