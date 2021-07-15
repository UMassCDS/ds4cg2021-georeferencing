import cv2
import matplotlib.pyplot as plt

# read images
img1 = cv2.imread('Images/Overlap1.jpg')
img2 = cv2.imread('Images/Overlap2.jpg')

# convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

edges1 = cv2.Canny(gray1, 100, 200)
edges2 = cv2.Canny(gray2, 100, 200)

# generate SIFT
sift = cv2.SIFT_create()

# generate keypoints
keypoints_1, descriptors_1 = sift.detectAndCompute(edges1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(edges2, None)

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(edges1, keypoints_1, edges2, keypoints_2, matches[:20], img2, flags=2)
plt.imshow(img3), plt.show()
