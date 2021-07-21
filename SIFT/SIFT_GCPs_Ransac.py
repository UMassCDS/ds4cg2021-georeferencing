import cv2
import matplotlib.pyplot as plt
import numpy as np
import GCPs as helper
import latlong as latlonghelper
from osgeo import gdal
import pandas as pd

if __name__ == '__main__':
    # read mac
    mac = cv2.imread('Images/mufs190-1952-dpb6h112-i001.reference.tif')
    mac_img = gdal.Open('Images/mufs190-1952-dpb6h112-i001.reference.tif')
    # storing the scaling ratio for mac
    x_scaled = 600
    y_scaled = 600
    mac_resize = [x_scaled / mac.shape[0], y_scaled / mac.shape[1]]
    # resizing mac
    mac = cv2.resize(mac, (x_scaled, y_scaled))

    # get the satellite tiles for this macconnell image
    df = pd.read_csv('Data/Mapping.csv')
    tiles = df.loc[df['MacFile'] == 'D:\\MacConnell\\Photos_Original\\b006\\mufs190-1952-dpb6h112-i001-001.tif'] \
    [['Tile1', 'Tile2', 'Tile3', 'Tile4', 'Tile5', 'Tile6', 'Tile7', 'Tile8', 'Tile9']]

    total, n = 0, 0
    for tile in tiles.values[0]:
        if tile is not None:
            # read mod
            mod = cv2.imread(tile)
            # store scaling ratio
            mod_resize = [x_scaled / mod.shape[0], y_scaled / mod.shape[1]]
            # resize mod
            mod = cv2.resize(mod, (x_scaled, y_scaled))

            # preprocess the images and generate keypoints from Harris Corner Detection
            cornersMac, cornersMod, keypointsMac, keypointsMod = helper.preprocess_images(mac, mod)

            # generate SIFT
            sift = cv2.SIFT_create()

            # generate keypoints and their descriptors
            descriptorsMac = sift.compute(cornersMac, keypointsMac)[1]
            descriptorsMod = sift.compute(cornersMod, keypointsMod)[1]

            # brute force feature matching of descriptors to match keypoints
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            #                              query            train
            matches = sorted(bf.match(descriptorsMac, descriptorsMod), key=lambda x: x.distance)
            top_matches = matches[:-int((len(matches) / 4) * 3)]

            # take all the points present in top_matches, find src and dist pts
            src_pts = np.float32([keypointsMac[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypointsMod[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # apply RANSAC
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # value is 1 for the inliers from RANSAC
            matchesMask = mask.ravel().tolist()

            # h, w = cornersMac.shape
            # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, M)
            # img2 = cv2.polylines(cornersMod, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # Reading the same images using gdal to obtain EPSG info
            mod_img = gdal.Open(tile, gdal.GA_ReadOnly)

            # For all the matches in ransac find the pixel to coord for mac and mod
            for i in range(0, len(matchesMask)):
                if matchesMask[i] == 1:
                    # scaling the pixel coordinates back to original sizes
                    scaled_src = [i / j for i, j in zip(src_pts[i][0], mac_resize)]
                    scaled_dst = [i / j for i, j in zip(dst_pts[i][0], mod_resize)]

                    # Converting the pixel to coordinates in corresponding systems
                    latmac, lonmac = latlonghelper.pixel2coord(mac_img, scaled_src[0], scaled_src[1])
                    latmod, lonmod = latlonghelper.pixel2coord(mod_img, scaled_dst[0], scaled_dst[1])

                    # Converting Modern image coordinates to Mac CRS
                    latmod, lonmod = latlonghelper.mod2maccoord(latmod, lonmod)

                    # Convert them into GPS CRS
                    latmac, lonmac = latlonghelper.coord2latlon(latmac, lonmac)
                    latmod, lonmod = latlonghelper.coord2latlon(latmod, lonmod)

                    # Calculate Distance
                    distance = latlonghelper.ground_dist(latmac, lonmac, latmod, lonmod)
                    total += distance
                    n += 1

            # draw_params = dict(matchesMask=matchesMask, flags=2)
            # img3 = cv2.drawMatches(mac, keypointsMac, mod, keypointsMod, matches, None, **draw_params)
            # plt.imshow(img3, 'gray'), plt.show()

    print(f'Average distance: {total / n} KMs')
