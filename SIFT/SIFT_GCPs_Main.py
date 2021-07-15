import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.cluster import OPTICS
import GCPs as helper

# read images
mac = cv2.imread('Images/River_Mac.jpg')
mod = cv2.imread('Images/River_Mod.jpg')

# preprocess the images
cornersMac, cornersMod, keypointsMac,keypointsMod = helper.preprocess_images(mac, mod)

# generate SIFT
sift = cv2.SIFT_create()

# generate their descriptors using sift

descriptorsMac = sift.compute(mac,keypointsMac)
descriptorsMod = sift.compute(mod,keypointsMod)

# brute force feature matching of descriptors to match keypoints
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
#                      query            train
matches = sorted(bf.match(descriptorsMac[1], descriptorsMod[1]), key=lambda x: x.distance)
#matches = matches[:-int((len(matches) / 4) * 3)]

'''# Visulaizing the matches from corners as keypoints
img3 = cv2.drawMatches(mac, keypointsMac, mod, keypointsMod, matches[:50], mod, flags=2)
plt.imshow(img3),plt.show()'''

# determine the length and slope of all 'lines' in keypoint matches
lines = helper.line_slopes_lengths(matches, keypointsMac, keypointsMod)

# fit the kmeans model
X = np.vstack([elem for elem in lines.values()])
optic = OPTICS(min_samples=5, p=1).fit(X)
labels = optic.fit_predict(X)

# find the cluster group sizes
clusters = Counter(labels)
# pop -1, the 'cluster' of points that dont belong to any true cluster
clusters.pop(-1, None)
# determine which points are in the cluster we want to map
matchesClustered = {lab: [] for lab in list(dict(sorted(clusters.items(), key=lambda item: item[1],
                                                        reverse=True)).keys())}
for target in matchesClustered.keys():
    for i, label in enumerate(labels):
        if label == target:
            matchesClustered[target].append(list(lines.keys())[i])

# extract matches to plot
bestCluster, score = helper.get_best_cluster(lines, matchesClustered)

# show images with keypoint matches
img3 = cv2.drawMatches(mac, keypointsMac, mod, keypointsMod, bestCluster, mac, flags=2)
# plt.figure(figsize=(50, 50))
print(len(bestCluster), score)
plt.imshow(img3), plt.show()
