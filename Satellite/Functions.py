import pyproj
import cv2
from skimage import exposure
import numpy as np
import yaml
from osgeo import gdal


def read_yaml(file_path):
    with open(file_path, "r") as fn:
        return yaml.safe_load(fn)


# takes a macconnell image and pixel coords and retuns the length in inches from the bottom left corner of the image
# that the pixel is
def pixel2inches(mac, x, y):
    x_new = x / mac.shape[0]
    y_new = y / mac.shape[1]
    xf = x_new * 9.0
    yf = (9.0 - (y_new * 9.0))
    return xf, yf


# function to translate pixel coordinates from a TIFF image to coords in the images epsg projection
def pixel2coord(img, col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    # unravel GDAL affine transform parameters
    c, a, b, f, d, e = img.GetGeoTransform()
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return xp, yp


def mod2mac(x1, y1, inpval):
    mac_coord = pyproj.Proj(projparams='epsg:26986')
    InputGrid = pyproj.Proj(projparams='epsg:' + str(inpval))
    return pyproj.transform(InputGrid, mac_coord, x1, y1)


# takes a psuedo-grayscale MacConnell image and a colored Modern image and returns the edge detected images after
# histogram matching the modern image to the MacConnel image and shifing both to grayscale
def preprocess_images(mac, mod):
    # histogram matching to adjust the modern image to match the pixel intensity of the MacConnell image
    multi = True if mod.shape[-1] > 1 else False

    matched_mod = exposure.match_histograms(mod, mac, multichannel=multi)

    # convert to grayscale
    gray_mac = cv2.cvtColor(mac, cv2.COLOR_BGR2GRAY)
    gray_mod = cv2.cvtColor(matched_mod, cv2.COLOR_BGR2GRAY)

    return gray_mac, gray_mod


# takes gray_scale images and returns the list of keypoints from the Harris Corner detection algorithm
def compute_harris_corner_keypoints(gray_mac, gray_mod):
    harris_mac = np.zeros(np.shape(gray_mac), dtype='uint8')
    harris_mod = np.zeros(np.shape(gray_mod), dtype='uint8')
    # perform harry corner detection
    corners_mac = cv2.cornerHarris(gray_mac, 2, 3, 0.04)
    corners_mod = cv2.cornerHarris(gray_mod, 2, 3, 0.04)
    harris_mac[corners_mac > 0.01 * corners_mac.max()] = 255
    harris_mod[corners_mod > 0.01 * corners_mod.max()] = 255

    mac_keypoints_idx = np.where(corners_mac > 0.01 * corners_mac.max())
    mac_keypoints = [cv2.KeyPoint(float(r), float(c), 10) for r, c in zip(mac_keypoints_idx[1], mac_keypoints_idx[0])]

    mod_keypoints_idx = np.where(corners_mod > 0.01 * corners_mod.max())
    mod_keypoints = [cv2.KeyPoint(float(r), float(c), 10) for r, c in zip(mod_keypoints_idx[1], mod_keypoints_idx[0])]
    return mac_keypoints, mod_keypoints


def cv_make_quadrants(img):
    # dividing the img using cv2
    quadrants = []
    width = img.shape[1]
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    width = img.shape[1]
    width_cutoff = width // 2
    l2 = img[:, :width_cutoff]
    l1 = img[:, width_cutoff:]
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r4 = img[:, :width_cutoff]
    r3 = img[:, width_cutoff:]
    # finish vertical devide image
    # rotate image to 90 COUNTERCLOCKWISE
    r4 = cv2.rotate(r4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotate image to 90 COUNTERCLOCKWISE
    r3 = cv2.rotate(r3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # save
    quadrants.append(l1)
    quadrants.append(r3)
    quadrants.append(l2)
    quadrants.append(r4)

    return quadrants


def cv_make_quadsofquads(img_quad):
    cv_img_oct = []
    for oct_i in img_quad:
        cv_img_oct.extend(cv_make_quadrants(oct_i))
    return cv_img_oct


def stitch_parts(cv_img_oct, top1, top2, bottom1, bottom2):
    img_h1 = cv2.hconcat([cv_img_oct[top1], cv_img_oct[top2]])
    img_h2 = cv2.hconcat([cv_img_oct[bottom1], cv_img_oct[bottom2]])

    return cv2.vconcat([img_h1, img_h2])


def stitchsatellite(names):
    vrt1 = gdal.BuildVRT("top.vrt", [names[0], names[1], names[2]])
    gdal.Translate("top.jp2", vrt1)

    vrt2 = gdal.BuildVRT("middle.vrt", [names[3], names[4], names[5]])
    gdal.Translate("middle.jp2", vrt2)

    vrt3 = gdal.BuildVRT("bottom.vrt", [names[6], names[7], names[8]])
    gdal.Translate("bottom.jp2", vrt3)

    top_files = ['top.jp2', 'middle.jp2', 'bottom.jp2']

    vrtd = gdal.BuildVRT(str("sat.vrt"), top_files)
    gdal.Translate(str("sat.jp2"), vrtd)

    return "sat.jp2"


def stitched_outputs(img):
    cv_img_quad = cv_make_quadrants(cv2.imread(img))

    cv_img_oct = cv_make_quadsofquads(cv_img_quad)

    list_cv = [cv_img_oct[0]]  # top left corner

    diag_top = stitch_parts(cv_img_oct, 1, 4, 3, 6)  # area between the top diagonals
    list_cv.append(diag_top)

    list_cv.append(cv_img_oct[5])  # top right corner

    diag_left = stitch_parts(cv_img_oct, 1, 4, 3, 6)  # Left area between diagonals
    list_cv.append(diag_left)

    center = stitch_parts(cv_img_oct, 3, 6, 9, 12)  # center image
    list_cv.append(center)

    diag_right = stitch_parts(cv_img_oct, 6, 7, 12, 13)  # right area between diagonals
    list_cv.append(diag_right)

    list_cv.append(cv_img_oct[15])  # bottom right corner

    diag_bottom = stitch_parts(cv_img_oct, 9, 12, 11, 14)  # bottom area between diagonals
    list_cv.append(diag_bottom)

    list_cv.append(cv_img_oct[10])  # bottom left corner

    return list_cv
