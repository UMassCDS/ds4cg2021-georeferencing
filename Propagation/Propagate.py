import os
import sys
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import Functions as helper
from osgeo import gdal, osr
from datetime import date
import logging


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    sr = osr.SpatialReference()
    # generate SIFT
    sift = cv2.SIFT_create()
    # brute force feature matching of descriptors to match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # read config and set parameters
    filenames = []
    refFn = None

    # get command line arguments
    try:
        refCode = sys.argv[1]
        ovCodes = sys.argv[2:]
    except IndexError as e:
        logging.error('Please provide a referenced image (argument 1) and at least one unreferenced image '
                      '(arguments 2 and futher).')
        sys.exit(1)

    mac = helper.read_yaml('config.yaml')['MACCONNELL']
    if mac['USE'] is True:
        logging.info('Using config for MacConnell dataset.')
        inp = mac['INPUT']
        out = mac['OUTPUT']

        unrefPath = inp['UNREF_PATH']
        refPath = inp['REF_PATH']
        initials = inp['INITIALS']
        maxGCPs = inp['MAX_GCPS']
        showMatches = inp['SHOW_MATCHES']
        GCPOut = out['GCP_OUTPUT_LOC']
        refOut = out['REF_OUTPUT_LOC']
        x_scaled = 600
        y_scaled = 600
        ransacThresh = 5
        DPI = 600
        blockSize = 2
        sobK = 3
        harK = 0.04
        targetEPSG = '26986'
        # set spatial reference
        sr.ImportFromEPSG(26986)

        # get filenames associated with each of the overlapping MacConnell images
        for code in ovCodes:
            GCP_fname = os.path.join(GCPOut, f'{code}_GCPs_{initials}_{date.today().strftime("%Y%m%d")}.txt')
            if os.path.exists(GCP_fname):
                logging.warning(f'{GCP_fname} already exists...Overwriting.')
            else:
                logging.info(f'{GCP_fname} created.')
            orig_fn = None
            found = False
            save = None
            # walk the directory to find the TIF file associated with this code
            for root, _, files in os.walk(unrefPath):
                for fn in files:
                    if fn[-3:] == 'tif' and code in fn.split('-'):
                        if found:
                            # found more than one TIFF file with this code
                            logging.error(f'Found more than one TIFF file with code {code} in folder {unrefPath}.')
                            sys.exit(1)
                        orig_fn = os.path.join(root, fn)
                        save = fn
                        found = True
            if orig_fn is None:
                # found no TIFF file with this code
                logging.error(f"Couldn't find an unreferenced MacConnell image with code {code} in folder {unrefPath}.")
                sys.exit(1)
            else:
                logging.info(f'Found unreferenced image {orig_fn}.')
                output_fn_temp = os.path.join(refOut, f'{save[:-4]}_temp.reference.tif')
                if os.path.exists(output_fn_temp.replace('_temp', '')):
                    logging.warning(f'{output_fn_temp.replace("_temp", "")} already exists...Overwriting.')
                filenames.append((orig_fn, output_fn_temp, GCP_fname))

        found = False
        for root, _, files in os.walk(refPath):
            for fn in files:
                if fn[-3:] == 'tif' and refCode in fn.split('-'):
                    if found:
                        logging.error(f'Found more than one TIFF file with code {refCode} in folder {refPath}.')
                        sys.exit(1)
                    refFn = os.path.join(root, fn)
                    found = True
        if refFn is None:
            logging.error(f"Couldn't find georeferenced MacConnell image with code {refCode} in folder {refPath}.")
            sys.exit(1)
        else:
            logging.info(f'Found referenced image {refFn}.')
        logging.info('Spatial reference set to EPSG:26986.')

    else:
        logging.info('Using config for other (not MacConnell) dataset.')
        inp = helper.read_yaml('config.yaml')['OTHER']['INPUT']
        out = helper.read_yaml('config.yaml')['OTHER']['OUTPUT']
        x_scaled = inp['X_SCALE']
        y_scaled = inp['Y_SCALE']
        showMatches = inp['SHOW_MATCHES']
        # set spatial reference
        targetEPSG = inp['EPSG']
        sr.ImportFromEPSG(targetEPSG)
        maxGCPs = inp['MAX_GCPS']
        DPI = inp['DPI']
        ransacThresh = inp['RANSAC_THRESHOLD']
        blockSize = inp['BLOCK_SIZE']
        sobK = inp['SOBEL_K']
        harK = inp['HARRIS_K']
        GCPOut = out['GCP_OUTPUT_LOC']
        refOut = out['REF_OUTPUT_LOC']
        refFn = refCode

        for path in ovCodes:
            GCP_fname = os.path.join(GCPOut,
                                     f'{os.path.split(path)[1][:-4]}_GCPs_{date.today().strftime("%Y%m%d")}.txt')
            if os.path.exists(GCP_fname):
                logging.warning(f'{GCP_fname} already exists...Overwriting.')
            else:
                logging.info(f'{GCP_fname} created.')

            output_fn_temp = os.path.join(refOut, f'{os.path.split(path)[1][:-4]}_temp.reference.tif')
            if os.path.exists(output_fn_temp.replace('_temp', '')):
                logging.warning(f'{output_fn_temp.replace("_temp", "")} already exists...Overwriting.')

            filenames.append((path, output_fn_temp, GCP_fname))

        logging.info(f'Spatial reference set to EPSG:{inp["EPSG"]}.')

    # read original referenced image
    refOG = cv2.imread(refFn)
    refGD = gdal.Open(refFn)
    # store scaling ratio
    ref_resize = [x_scaled / refOG.shape[0], y_scaled / refOG.shape[1]]
    # resize the images
    ref = cv2.resize(refOG, (x_scaled, y_scaled))

    if osr.SpatialReference(wkt=refGD.GetProjection()).GetAttrValue('AUTHORITY', 1) is None:
        logging.error(f'CRS from referenced image {refFn} could not be found. GCPs cannot be found from an '
                      f'image with no available CRS.')
        sys.exit(1)

    for orig_fn, output_fn_temp, GCP_fname in filenames:
        with open(GCP_fname, 'w') as f:
            # Create a copy of the original file and save it as the output filename:
            shutil.copy(orig_fn, output_fn_temp)
            logging.info(f'{output_fn_temp.replace("_temp", "")} created.')
            # Open the output file for writing for writing:
            # noinspection PyRedeclaration
            ds = gdal.Open(output_fn_temp, gdal.GA_Update)

            # read overlapping mac
            ovMac = cv2.imread(orig_fn)

            # store scaling ratio
            ovMac_resize = [x_scaled / ovMac.shape[0], y_scaled / ovMac.shape[1]]
            # resize the images
            ovMac = cv2.resize(ovMac, (x_scaled, y_scaled))

            # preprocess the images and generate keypoints from Harris Corner Detection
            grayRef, grayOvMac = helper.preprocess_images(ref, ovMac)
            logging.debug('Images preprocessed.')

            keypointsRef, keypointsOvMac = helper.compute_harris_corner_keypoints(grayRef, grayOvMac, blockSize, sobK,
                                                                                  harK)
            logging.debug('Keypoints detected with Harris Corner Detection.')

            # generate keypoints and their descriptors
            descriptorsRef = sift.compute(grayRef, keypointsRef)[1]
            descriptorsOvMac = sift.compute(grayOvMac, keypointsOvMac)[1]
            logging.debug('Descriptors computed with SIFT.')

            #                              query            train
            matches = sorted(bf.match(descriptorsRef, descriptorsOvMac), key=lambda x: x.distance)
            logging.debug('Matches found with brute force algorithm.')
            if len(matches) == 0:
                logging.error('No potential GCPs... Make sure the images are truly overlapping and have features to '
                              'match on.')
                sys.exit(1)

            # take all the points present in distributed_matches, find src and dist pts
            src_pts = np.float32([keypointsRef[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypointsOvMac[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # apply RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacThresh)
            logging.debug('Inliers calculated with RANSAC')
            # value is 1 for the inliers from RANSAC
            matchesMask = mask.ravel().tolist()
            top_matches = [match for i, match in enumerate(matches) if matchesMask[i] == 1]
            if not any(matchesMask):
                logging.error('None of the potential GCPs were found to be appropriate for georeferencing... '
                              'Make sure the images are truly overlapping and have features to match on.')
                sys.exit(1)
            else:
                logging.info(f'Found {len(matches)} total potential GCPs... Selecting {maxGCPs} evenly distributed '
                             f'GCPs from RANSAC\'s {len(top_matches)} inliers.')

            # select maxGCPs matches evenly distributed throughout the image
            distributed_matches = helper.select_matches(top_matches, keypointsOvMac, maxGCPs)
            logging.debug('Evenly distributed matches found.')
            # take all the points present in distributed_matches, find src and dist pts
            src_pts = np.float32([keypointsRef[m.queryIdx].pt for m in distributed_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypointsOvMac[m.trainIdx].pt for m in distributed_matches]).reshape(-1, 1, 2)

            # write all the selected matches and apply them to the unreferenced image
            gcps = []
            logging.info(f'Writing GCPs to {GCP_fname} and applying them to {output_fn_temp.replace("_temp", "")}...')
            for i in range(len(distributed_matches)):
                # scaling the pixel coordinates back to original sizes
                scaled_src = [k / j for k, j in zip(src_pts[i][0], ref_resize)]
                scaled_dst = [k / j for k, j in zip(dst_pts[i][0], ovMac_resize)]

                # converting the pixel to coordinates in corresponding systems
                CRSXRef, CRSYRef = helper.pixel2coord(refGD, scaled_src[0], scaled_src[1])

                # add the GCP
                gcps.append(gdal.GCP(CRSXRef, CRSYRef, 0, int(scaled_dst[0]), int(scaled_dst[1])))

                # calculate the inch coords for a GCP based on the pixel coords
                ovX, ovY = helper.pixel2inches(refOG, scaled_dst[0], scaled_dst[1], DPI)
                # write the GCPs to the text file
                ovX = format(np.round(ovX, 8), ".8f")
                ovY = format(np.round(ovY, 8), ".8f")
                f.write(f'{ovX}\t{ovY}\t{np.round(CRSXRef, 8)}\t{np.round(CRSYRef, 8)}\n')
            logging.debug('All GCPs written and applied.')

            if showMatches:
                img3 = cv2.drawMatches(ref, keypointsRef, ovMac, keypointsOvMac, distributed_matches, None, flags=2)
                plt.imshow(img3, 'gray'), plt.show()

        # apply the GCPs to the open output file
        output_fn = output_fn_temp.replace('_temp', '')
        srs = sr.ExportToWkt()
        ds.SetGCPs(gcps, sr.ExportToWkt())
        # ds = gdal.Translate(output_fn, ds, options=gdal.TranslateOptions(GCPs=gcps, outputSRS=sr.ExportToWkt(),
        #                                                                  resampleAlg='near'))
        gdal.Warp(output_fn, ds, options=gdal.WarpOptions(polynomialOrder=2, srcSRS=srs, dstSRS=srs))
        ds = None
        os.remove(output_fn_temp)
        logging.info(f'Completed {output_fn}!')
