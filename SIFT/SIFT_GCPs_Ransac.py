import cv2
import matplotlib.pyplot as plt
from geopy import distance as dst
import numpy as np
import GCPs as helper
import image_helper as imghelper
import latlong as latlonghelper
from osgeo import gdal
import pandas as pd


if __name__ == '__main__':
    # read mac
    mac = cv2.imread('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/mac_sample/mufs190-1952-dpb6h112-i001.reference.tif')
    mac_img = gdal.Open('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/mac_sample/bottom_1_gdal.tif')
    # get the quadrants for this image
    quads = imghelper.make_quadrants(mac)
    # define the new image scale
    x_scaled = 600
    y_scaled = 600

    # get the satellite tiles for this macconnell image
    df = pd.read_csv('Data/Mapping.csv')
    tiles = df.loc[df['MacFile'] == 'D:\\MacConnell\\Photos_Original\\b006\\mufs190-1952-dpb6h112-i001-001.tif'] \
        [['Tile1', 'Tile2', 'Tile3', 'Tile4', 'Tile5', 'Tile6', 'Tile7', 'Tile8', 'Tile9']]

    total, n = 0, 0
    for q, tile in enumerate(tiles.values[0]):
        # q == 0 & quads[0]         q == 3 & quads[1]       q == 6 & quads[2]           q == 8 & quads[3]
        if tile is not None and q == 0:
            # get corresponding quadrant
            mac = quads[0]
            # read mod
            mod = cv2.imread(tile)
            # store scaling ratio
            mac_resize = [x_scaled / mac.shape[0], y_scaled / mac.shape[1]]
            mod_resize = [x_scaled / mod.shape[0], y_scaled / mod.shape[1]]
            # resize the images
            mac = cv2.resize(mac, (x_scaled, y_scaled))
            mod = cv2.resize(mod, (x_scaled, y_scaled))

            # preprocess the images and generate keypoints from Harris Corner Detection
            grayMac, grayMod = helper.preprocess_images(mac, mod)
            keypointsMac, keypointsMod = helper.compute_harris_corner_keypoints(grayMac, grayMod)

            # generate SIFT
            sift = cv2.SIFT_create()

            # generate keypoints and their descriptors
            descriptorsMac = sift.compute(grayMac, keypointsMac)[1]
            descriptorsMod = sift.compute(grayMod, keypointsMod)[1]

            # brute force feature matching of descriptors to match keypoints
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            #                              query            train
            matches = sorted(bf.match(descriptorsMac, descriptorsMod), key=lambda x: x.distance)
            top_matches = matches[:-int((len(matches) / 4) * 3)]

            # take all the points present in top_matches, find src and dist pts
            src_pts = np.float32([keypointsMac[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypointsMod[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

            # apply RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # value is 1 for the inliers from RANSAC
            matchesMask = mask.ravel().tolist()

            # h, w = cornersMac.shape
            # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # trans = cv2.perspectiveTransform(pts, M)
            # img2 = cv2.polylines(cornersMod, [np.int32(trans)], True, 255, 3, cv2.LINE_AA)

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

                    # Converting Modern image coordinates to GPS CRS
                    latmod, lonmod = latlonghelper.modcoord2latlon(latmod, lonmod)

                    # Convert Mac coordinates into GPS CRS
                    latmac, lonmac = latlonghelper.maccoord2latlon(latmac, lonmac)

                    # Calculate Distance
                    total += dst.distance((latmac, lonmac), (latmod, lonmod)).m
                    n += 1

            draw_params = dict(matchesMask=matchesMask, flags=2)
            img3 = cv2.drawMatches(mac, keypointsMac, mod, keypointsMod, top_matches, None, **draw_params)
            plt.title(f'Average distance: {total / n} meters over {n} total matches')
            plt.imshow(img3, 'gray'), plt.show()
