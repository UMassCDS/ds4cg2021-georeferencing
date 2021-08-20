import os
import shutil
import cv2
import numpy as np
import Functions as helper
from osgeo import gdal, osr
from datetime import date
import sys
import logging
import pandas as pd
from zipfile import ZipFile
import wget


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    macCoords = pd.read_csv('Data/MacConnellCoords.csv')

    # get command line arguments
    try:
        code = sys.argv[1]
    except IndexError as e:
        logging.error('Please provide a referenced image (argument 1) and at least one unreferenced image '
                      '(arguments 2 and futher).')
        sys.exit(1)

    # Set spatial reference:
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(26986)
    # define the new image scale
    x_scaled = 600
    y_scaled = 600
    # generate SIFT
    sift = cv2.SIFT_create()
    # brute force feature matching of descriptors to match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # read config and set parameters
    mac = helper.read_yaml('config.yaml')
    inp = mac['INPUT']
    out = mac['OUTPUT']

    macPath = inp['MAC_PATH']
    satPath = inp['SAT_PATH']
    initials = inp['INITIALS']
    maxGCPs = inp['MAX_GCPS']
    avail_sat = inp['SAT_AVAILABLE']
    GCPOut = out['GCP_OUTPUT_LOC']
    refOut = out['REF_OUTPUT_LOC']

    GCP_fname = os.path.join(GCPOut, f'{code}_GCPs_{initials}_{date.today().strftime("%Y%m%d")}.txt')
    if os.path.exists(GCP_fname):
        logging.warning(f'{GCP_fname} already exists...Overwriting.')
    else:
        logging.info(f'{GCP_fname} created.')

    refFn = None
    orig_fn = None
    orig_fn_no_path = None
    found = False
    save = None
    # walk the directory to find the TIF file associated with this code
    for root, _, files in os.walk(macPath):
        for fn in files:
            if fn[-3:] == 'tif' and code in fn.split('-'):
                if found:
                    # found more than one TIFF file with this code
                    logging.error(f'Found more than one TIFF file with code {code} in folder {macPath}.')
                    sys.exit(1)
                orig_fn = os.path.join(root, fn)
                orig_fn_no_path = fn
                save = fn
                found = True
    if orig_fn is None:
        # found no TIFF file with this code
        logging.error(f"Couldn't find an unreferenced MacConnell image with code {code} in folder {macPath}.")
        sys.exit(1)
    else:
        logging.info(f'Found unreferenced image {orig_fn}.')
        output_fn_temp = os.path.join(refOut, f'{save[:-4]}_temp.reference.tif')
        if os.path.exists(output_fn_temp.replace('_temp', '')):
            logging.warning(f'{output_fn_temp.replace("_temp", "")} already exists...Overwriting.')

    if avail_sat:
        df = pd.read_csv('Data/Mapping.csv')
        tiles = df.loc[df['MacFile'] == orig_fn_no_path] \
            [['Tile1', 'Tile2', 'Tile3', 'Tile4', 'Tile5', 'Tile6', 'Tile7', 'Tile8', 'Tile9']]
        tiles = [os.path.join(satPath, os.path.join(os.path.split(os.path.split(tile)[0])[1], os.path.split(tile)[1]))
                 for tile in list(tiles.values[0])]

    else:
        # Downloading the Satellite tiles on the fly
        link = "http://download.massgis.digital.mass.gov/images/coq2019_15cm_jp2/"

        df = pd.read_csv('Data/Mapping.csv')
        tiles = df.loc[df['MacFile'] == orig_fn_no_path] \
            [['Tile1', 'Tile2', 'Tile3', 'Tile4', 'Tile5', 'Tile6', 'Tile7', 'Tile8', 'Tile9']]
        tiles = list(tiles.values[0])
        tile_nameslist = []
        for p in tiles:
            name = str(p)[-15:]
            name_str = name.replace('.jp2', '.zip')
            tile_nameslist.append(name_str)

        logging.info("Downloading the satellite tiles...")
        count = 0

        for file in tile_nameslist:
            file_name = file.split('.')[0]
            os.makedirs(os.path.join(satPath, file_name), exist_ok=True)
            wget.download(link + file, os.path.join(satPath, file_name, file))

            with ZipFile(os.path.join(satPath, file_name, file), 'r') as zipObj:
                zipObj.extractall(os.path.join(satPath, file_name))
            os.remove(os.path.join(satPath, file_name, file))
            tiles[count] = os.path.join(satPath, file_name, file.replace('.zip', '.jp2'))
            count += 1
        logging.info('Download complete.')

    logging.info('Spatial reference set to EPSG:26986.')

    # dividing the image to 9 parts
    list_cv = helper.stitched_outputs(orig_fn)

    # Merging the 9 satellite tiles into single tile
    logging.info(f'Merging the 9 satellite tiles to single tile.')
    merged_tile = helper.stitchsatellite(tiles)
    mod_img = gdal.Open(merged_tile, gdal.GA_ReadOnly)
    mod_width = mod_img.RasterXSize
    mod_height = mod_img.RasterXSize
    offset_dst_corners = [[0, 0], [mod_width / 3, 0], [2 * mod_width / 3, 0],
                          [0, mod_height / 3], [mod_width / 3, mod_height / 3], [2 * mod_width / 3, mod_height / 3],
                          [0, 2 * mod_height / 3], [mod_width / 3, 2 * mod_height / 3],
                          [2 * mod_width / 3, 2 * mod_height / 3]]

    mac_img = gdal.Open(orig_fn)
    mac_width = mac_img.RasterXSize
    mac_height = mac_img.RasterYSize
    # adjusting offsets
    offset_src_corners = [[0, 0], [mac_width / 4, 0], [3 * mac_width / 4, 0],
                          [0, mac_height / 4], [mac_width / 4, mac_height / 4], [mac_width / 2, mac_height / 4],
                          [0, 3 * mac_height / 4], [mac_width / 4, mac_height / 2],
                          [3 * mac_width / 4, 3 * mac_height / 4]]

    with open(GCP_fname, 'w') as f:
        # Create a copy of the original file and save it as the output filename:
        shutil.copy(orig_fn, output_fn_temp)
        # Open the output file for writing
        ds = gdal.Open(output_fn_temp, gdal.GA_Update)
        logging.info(f'{output_fn_temp.replace("_temp", "")} created.')

        # creating empty lists to combine matches across all combinations of Macconnell to satellite
        combined_matches = []
        combined_src = []
        combined_dst = []

        total, n = 0, 0
        for q, tile in enumerate(tiles):
            if tile is not None:
                logging.info(f'Started tile {q + 1}...')
                # get corresponding quadrant
                mac = list_cv[q]

                # read mod
                mod = cv2.imread(tile)
                # store scaling ratio
                mac_resize = [x_scaled / mac.shape[0], y_scaled / mac.shape[1]]
                mod_resize = [x_scaled / mod.shape[0], y_scaled / mod.shape[1]]

                # resize the images
                mac = cv2.resize(mac, (x_scaled, y_scaled))
                mod = cv2.resize(mod, (x_scaled, y_scaled))

                # preprocess the images and
                grayMac, grayMod = helper.preprocess_images(mac, mod)
                logging.debug('Images preprocessed.')

                # generate keypoints from Harris Corner Detection
                keypointsMac, keypointsMod = helper.compute_harris_corner_keypoints(grayMac, grayMod)
                logging.debug('Keypoints detected with Harris Corner Detection.')

                # generate keypoints and their descriptors
                descriptorsMac = sift.compute(grayMac, keypointsMac)[1]
                descriptorsMod = sift.compute(grayMod, keypointsMod)[1]
                logging.debug('Descriptors computed with SIFT.')

                #                              query            train
                matches = sorted(bf.match(descriptorsMac, descriptorsMod), key=lambda x: x.distance)
                top_matches = matches[:-int((len(matches) / 4) * 3)]
                combined_matches.extend(top_matches)

                # take all the points present in top_matches, find src and dest pts
                src_pts = np.float32([keypointsMac[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypointsMod[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

                # apply RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # value is 1 for the inliers from RANSAC
                matchesMask = mask.ravel().tolist()

                scaled_src = []
                scaled_dst = []
                index = []
                _id = 0
                for i in range(0, len(src_pts)):
                    if _id == maxGCPs:
                        break
                    if matchesMask[i] == 1:
                        _id += 1
                        index.append(i)
                        scaled_src.append([float(int(i / j) + x) for i, j, x in
                                           zip(src_pts[i][0], mac_resize, offset_src_corners[q])])
                        scaled_dst.append([float(int(i / j) + x) for i, j, x in
                                           zip(dst_pts[i][0], mod_resize, offset_dst_corners[q])])

                _id = 0
                for m in index:
                    keypointsMac[top_matches[m].queryIdx].pt = (scaled_src[_id][0], scaled_src[_id][1])
                    keypointsMod[top_matches[m].trainIdx].pt = (scaled_dst[_id][0], scaled_dst[_id][1])
                    _id += 1

                combined_src.extend(scaled_src)
                combined_dst.extend(scaled_dst)

        if len(combined_matches) == 0:
            logging.error(
                'No potential GCPs... Make sure the images are of the same area, but even so, they may not be '
                'fit for automatic georeferencing.')
            sys.exit(1)
        else:
            logging.info(f'Found {len(combined_matches)} total potential GCPs... Selecting the {maxGCPs} best from '
                         f'each of the 9 tiles.')

        top_matches = sorted(combined_matches, key=lambda x: x.distance)

        final_src = combined_src
        final_dst = combined_dst

        gcps = []
        logging.info(f'Writing GCPs to {GCP_fname} and applying them to {output_fn_temp.replace("_temp", "")}...')
        prj = mod_img.GetProjection()
        srs = osr.SpatialReference(wkt=prj)

        # For all the matches in ransac find the pixel to coord for mac and mod
        for i in range(0, len(final_src)):
            # Converting the pixel to coordinates in corresponding CRS for modern img
            CRSXmod, CRSYmod = helper.pixel2coord(mod_img, final_dst[i][0], final_dst[i][1])

            # Converting Modern image coordinates to Maconell Image CRS
            CRSXmac, CRSYmac = helper.mod2mac(CRSXmod, CRSYmod, srs.GetAttrValue('AUTHORITY', 1))

            # add the GCP
            gcps.append(gdal.GCP(CRSXmac, CRSYmac, 0, int(final_src[i][0]), int(final_src[i][1])))

            # calculate the inch coords for a GCP based on the pixel coords
            ovX, ovY = helper.pixel2inches(cv2.imread(orig_fn), final_src[i][0], final_src[i][1])
            # write the GCPs to the text file
            ovX = format(np.round(ovX, 8), ".8f")
            ovY = format(np.round(ovY, 8), ".8f")

            f.write(f'{ovX}\t{ovY}\t{np.round(CRSXmac, 8)}\t{np.round(CRSYmac, 8)}\n')
        logging.debug('All GCPs written and applied.')

    # Apply the GCPs to the open output file:
    ds.SetGCPs(gcps, sr.ExportToWkt())
    srs = sr.ExportToWkt()
    output_fn = output_fn_temp.replace('_temp', '')
    gdal.Warp(output_fn, ds, options=gdal.WarpOptions(polynomialOrder=2, srcSRS=srs, dstSRS=srs))
    ds = None
    mod_img = None
    for fn in ['bottom.jp2', 'bottom.vrt', 'middle.jp2', 'middle.vrt', 'top.jp2', 'top.vrt', 'sat.jp2', 'sat.vrt']:
        os.remove(fn)
    os.remove(output_fn_temp)
    logging.info('Completed!')
