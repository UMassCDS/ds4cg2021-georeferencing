import cv2
import pandas as pd
from geopy import distance as dst
import numpy as np
import GCPs as helper
import latlong as latlonghelper
from osgeo import gdal, osr


if __name__ == '__main__':
    # generate SIFT
    sift = cv2.SIFT_create()
    # brute force feature matching of descriptors to match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    with open('Data\\PropagationDistances.csv', 'w') as f:
        f.write('Center,Tile1,PtDistance1,GCPDistance1,Tile2,PtDistance2,GCPDistance2,Tile3,PtDistance3,GCPDistance3,'
                'Tile4,PtDistance4,GCPDistance4,Tile5,PtDistance5,GCPDistance5,Tile6,PtDistance6,GCPDistance6,'
                'Tile7,PtDistance7,GCPDistance7,Tile8,PtDistance8,GCPDistance8\n')
        mapping = pd.read_csv('Data\\MacConnellMapping.csv')
        c = 0
        for _, row in mapping.iterrows():
            lst = [(row['Tile1'], row['Distance1']), (row['Tile2'], row['Distance2']), (row['Tile3'], row['Distance3']),
                   (row['Tile4'], row['Distance4']), (row['Tile5'], row['Distance5']), (row['Tile6'], row['Distance6']),
                   (row['Tile7'], row['Distance7']), (row['Tile8'], row['Distance8'])]
            REFPATH = row['Center']
            f.write(f'{REFPATH}')
            X_SCALE = 600
            Y_SCALE = 600

            # Set spatial reference:
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(4326)
            # read mac
            refOG = cv2.imread(REFPATH)
            ref_resize = [X_SCALE / refOG.shape[0], Y_SCALE / refOG.shape[1]]
            ref = cv2.resize(refOG, (X_SCALE, Y_SCALE))

            for elem in lst:
                if elem[0] == 'None':
                    f.write(f',{elem[0]},-1,-1')
                else:
                    OVPATH = elem[0]
                    f.write(f',{OVPATH},{elem[1]}')
                    # read overlapping mac
                    ovMac = cv2.imread(OVPATH)

                    # store scaling ratio
                    ovMac_resize = [X_SCALE / ovMac.shape[0], Y_SCALE / ovMac.shape[1]]
                    # resize the images
                    ovMac = cv2.resize(ovMac, (X_SCALE, Y_SCALE))

                    # preprocess the images and generate keypoints from Harris Corner Detection
                    grayRef, grayOvMac = helper.preprocess_images(ref, ovMac)
                    keypointsRef, keypointsOvMac = helper.compute_harris_corner_keypoints(grayRef, grayOvMac)

                    # generate keypoints and their descriptors
                    descriptorsRef = sift.compute(grayRef, keypointsRef)[1]
                    descriptorsOvMac = sift.compute(grayOvMac, keypointsOvMac)[1]

                    #                              query            train
                    matches = sorted(bf.match(descriptorsRef, descriptorsOvMac), key=lambda x: x.distance)
                    top_matches = matches[:-int((len(matches) / 4) * 3)]

                    # take all the points present in top_matches, find src and dist pts
                    src_pts = np.float32([keypointsRef[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypointsOvMac[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

                    # apply RANSAC
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    # value is 1 for the inliers from RANSAC
                    matchesMask = mask.ravel().tolist()

                    # For all the matches in ransac find the pixel to coord for both images
                    total, n = 0, 0
                    for i in range(0, len(matchesMask)):
                        if matchesMask[i] == 1:
                            # scaling the pixel coordinates back to original sizes
                            scaled_src = [i / j for i, j in zip(src_pts[i][0], ref_resize)]
                            scaled_dst = [i / j for i, j in zip(dst_pts[i][0], ovMac_resize)]

                            # converting the pixel to coordinates in corresponding systems
                            CRSXRef, CRSYRef = latlonghelper.pixel2coord(gdal.Open(REFPATH), scaled_src[0],
                                                                         scaled_src[1])
                            CRSXOvMac, CRSYOvMac = latlonghelper.pixel2coord(gdal.Open(OVPATH), scaled_dst[0],
                                                                             scaled_dst[1])

                            # convert them into GPS CRS
                            latRef, lonRef = latlonghelper.mac2latlon(CRSXRef, CRSYRef)
                            latOvMac, lonOvMac = latlonghelper.mac2latlon(CRSXOvMac, CRSYOvMac)

                            # calculate distance
                            total += dst.distance((latRef, lonRef), (latOvMac, lonOvMac)).m
                            n += 1

                    f.write(f',{total / n}')
            f.write('\n')
            c += 1
            print(f'{c / len(mapping["Center"]) * 100}% done')
