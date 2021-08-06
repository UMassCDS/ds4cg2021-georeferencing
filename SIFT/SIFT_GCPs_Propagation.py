import shutil
import cv2
from geopy import distance as dst
import matplotlib.pyplot as plt
import numpy as np
import GCPs as helper
import latlong as latlonghelper
from osgeo import gdal, osr
import image_helper as imagehelper
from datetime import date
import yaml


def read_yaml(file_path):
    with open(file_path, "r") as fn:
        return yaml.safe_load(fn)


if __name__ == '__main__':
    # read config and set parameters
    inp = read_yaml('config.yaml')['INPUT']
    out = read_yaml('config.yaml')['OUTPUT']

    refPath = inp['REFPATH']
    ovPath = inp['OVPATH']
    stub = inp['STUB']
    initials = inp['INITIALS']
    x_scaled = inp['X_SCALE']
    y_scaled = inp['Y_SCALE']

    GCP_fname = f'D:\\MacConnell\\Propagation\\{stub}_GCPs_{initials}_{date.today().strftime("%Y%m%d")}.txt'

    orig_fn = out['ORIG_FN']
    output_fn = out['OUTPUT_FN']
    EPSG = out['EPSG']

    with open(GCP_fname, 'w') as f:
        # Create a copy of the original file and save it as the output filename:
        shutil.copy(orig_fn, output_fn)
        # Open the output file for writing for writing:
        ds = gdal.Open(output_fn, gdal.GA_Update)
        # Set spatial reference:
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(EPSG)

        # read mac
        refOG = cv2.imread(refPath)
        # read overlapping mac
        ovMac = cv2.imread(ovPath)

        # store scaling ratio
        ref_resize = [x_scaled / refOG.shape[0], y_scaled / refOG.shape[1]]
        ovMac_resize = [x_scaled / ovMac.shape[0], y_scaled / ovMac.shape[1]]
        # resize the images
        ref = cv2.resize(refOG, (x_scaled, y_scaled))
        ovMac = cv2.resize(ovMac, (x_scaled, y_scaled))

        # preprocess the images and generate keypoints from Harris Corner Detection
        grayRef, grayOvMac = helper.preprocess_images(ref, ovMac)
        keypointsRef, keypointsOvMac = helper.compute_harris_corner_keypoints(grayRef, grayOvMac)

        # generate SIFT
        sift = cv2.SIFT_create()

        # generate keypoints and their descriptors
        descriptorsRef = sift.compute(grayRef, keypointsRef)[1]
        descriptorsOvMac = sift.compute(grayOvMac, keypointsOvMac)[1]

        # brute force feature matching of descriptors to match keypoints
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
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
        gcps = []
        for i in range(0, len(matchesMask)):
            if matchesMask[i] == 1:
                # scaling the pixel coordinates back to original sizes
                scaled_src = [i / j for i, j in zip(src_pts[i][0], ref_resize)]
                scaled_dst = [i / j for i, j in zip(dst_pts[i][0], ovMac_resize)]

                # converting the pixel to coordinates in corresponding systems
                CRSXRef, CRSYRef = latlonghelper.pixel2coord(gdal.Open(refPath), scaled_src[0], scaled_src[1])
                CRSXOvMac, CRSYOvMac = latlonghelper.pixel2coord(gdal.Open(ovPath), scaled_dst[0], scaled_dst[1])

                # convert them into GPS CRS
                latRef, lonRef = latlonghelper.mac2latlon(CRSXRef, CRSYRef)
                latOvMac, lonOvMac = latlonghelper.mac2latlon(CRSXOvMac, CRSYOvMac)

                # calculate distance
                total += dst.distance((latRef, lonRef), (latOvMac, lonOvMac)).m
                n += 1

                # add the GCP
                gcps.append(gdal.GCP(lonRef, latRef, 0, int(scaled_dst[0]), int(scaled_dst[1])))

                # calculate the inch coords for a GCP based on the pixel coords
                ovX, ovY = imagehelper.pixel2inches(refOG, scaled_dst[0], scaled_dst[1])
                # write the GCPs to the text file
                ovX = format(np.round(ovX, 8), ".8f")
                ovY = format(np.round(ovY, 8), ".8f")
                f.write(f'{ovX}\t{ovY}\t{np.round(CRSXRef, 8)}\t{np.round(CRSYRef, 8)}\n')

        draw_params = dict(matchesMask=matchesMask, flags=2)
        img3 = cv2.drawMatches(ref, keypointsRef, ovMac, keypointsOvMac, top_matches, None, **draw_params)
        plt.title(f'Average distance: {total / n} meters over {n} total matches')
        plt.imshow(img3, 'gray'), plt.show()

    # Apply the GCPs to the open output file:
    ds.SetGCPs(gcps, sr.ExportToWkt())

    # Close the output file in order to be able to work with it in other programs:
    ds = None
