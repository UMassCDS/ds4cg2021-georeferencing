import os
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Functions as helper
from osgeo import gdal, osr
from datetime import date
import sys

# TODO: Search entire directory recursively to find the right files
# TODO: Implement logger
# TODO: Take 25 matches from across the image
if __name__ == '__main__':
    macCoords = pd.read_csv('Data/MacConnellCoords.csv')
    # Set spatial reference:
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    # generate SIFT
    sift = cv2.SIFT_create()
    # brute force feature matching of descriptors to match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    filenames = []
    refFn = None
    x_scaled = 600
    y_scaled = 600
    maxGCPs = 25
    # read config and set parameters
    mac = helper.read_yaml('config.yaml')['MACCONNELL']
    if mac['USE'] is True:
        inp = mac['INPUT']
        out = mac['OUTPUT']

        refCode = inp['REF_CODE']
        ovCodes = [code.strip() for code in inp['OVERLAPPING_CODES'].split(',')]
        unrefPath = inp['UNREF_PATH']
        refPath = inp['REF_PATH']
        satPath = inp['SAT_PATH']
        initials = inp['INITIALS']
        maxGCPs = inp['MAX_GCPS']
        GCPOut = out['GCP_OUTPUT_LOC']
        refOut = out['REF_OUTPUT_LOC']

        for code in ovCodes:
            GCP_fname = f'{GCPOut}{code}_GCPs_{initials}_{date.today().strftime("%Y%m%d")}.txt'
            orig_fn = None
            found = False
            for fn in macCoords['Filename']:
                if code == fn.split('\\')[-1].split('-')[-3] and fn.split('\\')[-1][-3:] == 'tif':
                    temp = fn[30:].replace('\\', '/')
                    orig_fn = f'{unrefPath}{temp}'
                    if found:
                        print(f'Found more than one TIFF file with code {code} in folder {unrefPath}.')
                        sys.exit(1)
                    found = True
            if orig_fn is None:
                print(f'Could not find an unreferenced MacConnell image with code {code} in folder {unrefPath}.')
                sys.exit(1)
            output_fn = f'{refOut}{orig_fn.split("/")[-1][:-4]}.reference.tif'
            filenames.append((orig_fn, output_fn, GCP_fname))

        found = False
        for fn in os.listdir(refPath):
            if refCode == fn.split('-')[-3] and fn[-3:] == 'tif':
                refFn = refPath + fn
                if found:
                    print(f'Found more than one TIFF file with code {refCode} in folder {refPath}.')
                    sys.exit(1)
                found = True
        if refFn is None:
            print(f'Could not find georeferenced MacConnell image with code {refCode} in folder {refPath}.')
            sys.exit(1)

    else:
        inp = helper.read_yaml('config.yaml')['OTHER']['INPUT']
        out = helper.read_yaml('config.yaml')['OTHER']['OUTPUT']

    # read original referenced image
    refOG = cv2.imread(refFn)
    # store scaling ratio
    ref_resize = [x_scaled / refOG.shape[0], y_scaled / refOG.shape[1]]
    # resize the images
    ref = cv2.resize(refOG, (x_scaled, y_scaled))

    for orig_fn, output_fn, GCP_fname in filenames:
        with open(GCP_fname, 'w') as f:
            # Create a copy of the original file and save it as the output filename:
            shutil.copy(orig_fn, output_fn)
            # Open the output file for writing for writing:
            ds = gdal.Open(output_fn, gdal.GA_Update)

            # read overlapping mac
            ovMac = cv2.imread(orig_fn)

            # store scaling ratio
            ovMac_resize = [x_scaled / ovMac.shape[0], y_scaled / ovMac.shape[1]]
            # resize the images
            ovMac = cv2.resize(ovMac, (x_scaled, y_scaled))

            # preprocess the images and generate keypoints from Harris Corner Detection
            grayRef, grayOvMac = helper.preprocess_images(ref, ovMac)
            keypointsRef, keypointsOvMac = helper.compute_harris_corner_keypoints(grayRef, grayOvMac)

            # generate keypoints and their descriptors
            descriptorsRef = sift.compute(grayRef, keypointsRef)[1]
            descriptorsOvMac = sift.compute(grayOvMac, keypointsOvMac)[1]

            #                                  query            train
            top_matches = sorted(bf.match(descriptorsRef, descriptorsOvMac), key=lambda x: x.distance)
            # top_matches = matches[:maxGCPs]
            # top_matches = matches[:-int((len(matches) / 4) * 3)]

            # take all the points present in top_matches, find src and dist pts
            src_pts = np.float32([keypointsRef[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypointsOvMac[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

            # apply RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # value is 1 for the inliers from RANSAC
            matchesMask = mask.ravel().tolist()

            # For all the matches in ransac find the pixel to coord for both images
            gcps = []
            for i in range(0, len(matchesMask)):
                if matchesMask[i] == 1:
                    # scaling the pixel coordinates back to original sizes
                    scaled_src = [i / j for i, j in zip(src_pts[i][0], ref_resize)]
                    scaled_dst = [i / j for i, j in zip(dst_pts[i][0], ovMac_resize)]

                    # converting the pixel to coordinates in corresponding systems
                    CRSXRef, CRSYRef = helper.pixel2coord(gdal.Open(refFn), scaled_src[0], scaled_src[1])

                    # convert them into GPS CRS
                    latRef, lonRef = helper.mac2latlon(CRSXRef, CRSYRef)

                    # add the GCP
                    gcps.append(gdal.GCP(lonRef, latRef, 0, int(scaled_dst[0]), int(scaled_dst[1])))

                    # calculate the inch coords for a GCP based on the pixel coords
                    ovX, ovY = helper.pixel2inches(refOG, scaled_dst[0], scaled_dst[1])
                    # write the GCPs to the text file
                    ovX = format(np.round(ovX, 8), ".8f")
                    ovY = format(np.round(ovY, 8), ".8f")
                    f.write(f'{ovX}\t{ovY}\t{np.round(CRSXRef, 8)}\t{np.round(CRSYRef, 8)}\n')

            draw_params = dict(matchesMask=matchesMask, flags=2)
            img3 = cv2.drawMatches(ref, keypointsRef, ovMac, keypointsOvMac, top_matches, None, **draw_params)
            plt.imshow(img3, 'gray'), plt.show()

        # Apply the GCPs to the open output file:
        ds.SetGCPs(gcps, sr.ExportToWkt())
