import sys
from collections import Counter
import numpy as np
from skimage import exposure
import cv2
from sklearn.cluster import OPTICS


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


# takes the list of keypoint matches, and the lists of all keypoints and returns a dictionary {match: (length, slope)}
def line_slopes_lengths(matches, keypointsMac, keypointsMod):
    lines = {}
    for match in matches:
        query = match.queryIdx
        train = match.trainIdx
        # calculate length and slope
        length = np.linalg.norm(np.array(keypointsMac[query].pt) -
                                np.array(keypointsMod[train].pt))
        if keypointsMac[query].pt[0] - keypointsMod[train].pt[0] == 0:
            continue
        else:
            slope = (keypointsMac[query].pt[1] - keypointsMod[train].pt[1]) / \
                    (keypointsMac[query].pt[0] - keypointsMod[train].pt[0])
        lines[match] = (length, slope)
    return lines


# takes the list of keypoint matches, and the lists of all keypoints and returns a dictionary {match: (length, slope)}
def get_line_slopes_lengths(matches, keypointsMac, keypointsMod):
    lines = {}
    for match in matches:
        query = match.queryIdx
        train = match.trainIdx
        # calculate length and slope
        length = np.linalg.norm(np.array(keypointsMac[query].pt) -
                                np.array(keypointsMod[train].pt))
        slope = (keypointsMac[query].pt[1] - keypointsMod[train].pt[1]) / \
                (keypointsMac[query].pt[0] - keypointsMod[train].pt[0])
        lines[match] = (length, slope)
    return lines


# takes the dictionary of 'keypoint match: (length, slope)' and returns a dictionary 'cluster label: keypoint match'
def get_match_clusters(lines):
    # fit the kmeans model
    X = np.vstack([elem for elem in lines.values()])
    optic = OPTICS(min_samples=5, p=1).fit(X)
    labels = optic.fit_predict(X)

    # find the cluster group sizes
    clusters = Counter(labels)
    # pop -1, the 'cluster' of points that dont belong to any true cluster
    clusters.pop(-1, None)
    # determine which points are in the cluster we want to map
    matchesClustered = {lab: [] for lab in list(dict(sorted(clusters.items(), key=lambda item: item[1],
                                                            reverse=True)).keys())}
    for target in matchesClustered.keys():
        for i, label in enumerate(labels):
            if label == target:
                matchesClustered[target].append(list(lines.keys())[i])

    return matchesClustered


# takes a dictionary of 'cluster label: keypoint matches' and the corresponding (distance, slope)
# and returns the cluster of match objects with the lowest average distance between them
def get_best_cluster(lines, clusters):
    best_avg_dist = sys.maxsize
    best_label = -1
    for label, matches in clusters.items():
        total = 0
        for match in matches:
            total += np.linalg.norm(lines[match])
        avg_dist = total / len(matches)
        if best_avg_dist > avg_dist:
            best_avg_dist = avg_dist
            best_label = label
    return clusters[best_label], best_avg_dist
