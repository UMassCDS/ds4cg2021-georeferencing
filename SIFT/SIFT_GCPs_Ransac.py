import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.cluster import OPTICS
import GCPs as helper
import time
import image_helper as imghelper
import latlong as latlonghelper
from osgeo import gdal

# read images
mac_original = cv2.imread('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/mac_sample/mufs190-1952-dpb6h112-i001.reference.tif')
mod = cv2.imread('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/sat_sample/18TXM990890/18TXM990890.jp2') 

quad = imghelper.make_quadrants(mac_original)
mac = quad[0]
# storing the scaling ratio
mac_resize = [600 / mac.shape[0], 600 / mac.shape[1]]
mod_resize = [600 / mod.shape[0], 600 / mod.shape[1]]

# resizing the images
mac = cv2.resize(mac, (600, 600))
mod = cv2.resize(mod, (600, 600))

# preprocess the images
cornersMac, cornersMod, keypointsMac, keypointsMod = helper.preprocess_images(mac, mod)

# generate SIFT
sift = cv2.SIFT_create()

# generate keypoints and their descriptors
descriptorsMac = sift.compute(cornersMac, keypointsMac)
descriptorsMod = sift.compute(cornersMod, keypointsMod)

# brute force feature matching of descriptors to match keypoints
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
#                      query            train
matches = sorted(bf.match(descriptorsMac[1], descriptorsMod[1]), key=lambda x: x.distance)

top_matches = matches[:-int((len(matches) / 4) * 3)]
print(len(top_matches))

# take all the points present in top_matches, find src, dist pts and 
src_pts = np.float32([keypointsMac[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypointsMod[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# value is 1 for the inliers from RANSAC
matchesMask = mask.ravel().tolist()

h, w = cornersMac.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

img2 = cv2.polylines(cornersMod, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

# Reading the same images using gdal to obtain EPSG info
mac_img= gdal.Open('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/mac_sample/mufs190-1952-dpb6h112-i001.reference.tif')
mod_img = gdal.Open('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/sat_sample/18TXM990890/18TXM990890.jp2',gdal.GA_ReadOnly)

# For all the matches in ransac find the pixel to coord for mac and mod
for i in range(0, len(matchesMask)):
    if matchesMask[i] == 1:
        # scaling the pixel co ordinates back to original sizes
        scaled_src = [i / j for i, j in zip(src_pts[i][0], mac_resize)]
        scaled_dst = [i / j for i, j in zip(dst_pts[i][0], mod_resize)]

        # Converting the pixel to coordinates in corresponding systems
        latmac, lonmac = latlonghelper.pixel2coord(mac_img, scaled_src[0], scaled_src[1])
        latmod, lonmod = latlonghelper.pixel2coord(mod_img, scaled_dst[0], scaled_dst[1])

        # Converting Modern image coordinates to Mac CRS
        latmod, lonmod = latlonghelper.mod2maccoord(latmod, lonmod)
        print(latmac, lonmac)
        print(latmod, lonmod)

        # Convert them into GPS CRS
        latmac, lonmac = latlonghelper.coord2latlon(latmac, lonmac)
        latmod, lonmod = latlonghelper.coord2latlon(latmod, lonmod)
        # Calculate Distance
        latlonghelper.ground_dist(latmac, lonmac, latmod, lonmod)
        print("-----")

draw_params = dict(matchesMask=matchesMask, flags=2)

img3 = cv2.drawMatches(mac, keypointsMac, mod, keypointsMod, top_matches, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()
