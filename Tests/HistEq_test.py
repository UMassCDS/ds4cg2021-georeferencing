import cv2
import matplotlib.pyplot as plt
from skimage import exposure

# load the image
mac = cv2.imread('Images/River_Mac.jpg')
mod = cv2.imread('Images/River_Mod.jpg')

# histogram matching
multi = True if mod.shape[-1] > 1 else False
matchedMod = exposure.match_histograms(mod, mac, multichannel=multi)

# convert to grayscale
grayMac = cv2.cvtColor(mac, cv2.COLOR_BGR2GRAY)
grayMod = cv2.cvtColor(matchedMod, cv2.COLOR_BGR2GRAY)

# plot the images
plt.subplot(121), plt.imshow(grayMac, cmap='gray')
plt.title('Gray MacConnell Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(grayMod, cmap='gray')
plt.title('Histogram Matched Modern Image'), plt.xticks([]), plt.yticks([])
plt.show()
