import cv2
from matplotlib import pyplot as plt
from skimage import exposure

# read images
mod = cv2.imread('Images/Boston_Mod.jpg')
mac = cv2.imread('Images/Boston_Mac.jpg')

# histogram matching to adjust the modern image to match the pixel intensity of the MacConnell image
multi = True if mod.shape[-1] > 1 else False
matchedMod = exposure.match_histograms(mod, mac, multichannel=multi)

# convert to grayscale
grayMod = cv2.cvtColor(mod, cv2.COLOR_BGR2GRAY)
grayMac = cv2.cvtColor(mac, cv2.COLOR_BGR2GRAY)

# edge detection using Canny
edgesMac = cv2.Canny(grayMac, 100, 200)
edgesMod = cv2.Canny(matchedMod, 100, 200)

# create FREAK and SIFT instance
freakExtractor = cv2.xfeatures2d.FREAK_create()
sift = cv2.SIFT_create()

# generate keypoints with SIFT and their descriptors with FREAK
keypointsMac, _ = sift.detectAndCompute(grayMac, None)
keypointsMod, _ = sift.detectAndCompute(grayMod, None)
descriptorsMac = freakExtractor.compute(grayMac, keypointsMac)
descriptorsMod = freakExtractor.compute(grayMod, keypointsMod)
print(descriptorsMac[-1], descriptorsMod[-1])
# brute force feature matching of descriptors to match keypoints
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
#                      query            train
matches = bf.match(descriptorsMac[-1], descriptorsMod[-1])
matches = sorted(matches, key=lambda x: x.distance)

# plot matches
img3 = cv2.drawMatchesKnn(mac, keypointsMac, mod, keypointsMod, matches, None, flags=2)
plt.imshow(img3)
plt.show()
