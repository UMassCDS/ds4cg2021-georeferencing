import cv2
import matplotlib.pyplot as plt
import GeoReferencing.SIFT.GCPs as helper

if __name__ == '__main__':
    # read images
    mac = cv2.imread('Images/River_Mac.jpg')
    mod = cv2.imread('Images/River_Mod.jpg')

    # preprocess the images
    edgesMac, edgesMod = helper.preprocess_images(mac, mod)

    # generate SIFT
    sift = cv2.SIFT_create()

    # generate keypoints and their descriptors
    keypointsMac, descriptorsMac = sift.detectAndCompute(edgesMac, None)
    keypointsMod, descriptorsMod = sift.detectAndCompute(edgesMod, None)

    # brute force feature matching of descriptors to match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    #                            query            train
    matches = sorted(bf.match(descriptorsMac, descriptorsMod), key=lambda x: x.distance)
    # threshold to select the top 25% of matches by Euclidean distance between descriptors
    # matches = matches[:-int((len(matches) / 4) * 3)]

    # determine the length and slope of all 'lines' in keypoint matches
    lines = helper.get_line_slopes_lengths(matches, keypointsMac, keypointsMod)

    # cluster the matches by length and slope
    matchesClustered = helper.get_match_clusters(lines)
    bestCluster = helper.get_best_cluster(lines, matchesClustered)

    # show images with keypoint matches
    img3 = cv2.drawMatches(mac, keypointsMac, mod, keypointsMod, bestCluster, mac, flags=2)
    # plt.figure(figsize=(50, 50))
    plt.imshow(img3), plt.show()
