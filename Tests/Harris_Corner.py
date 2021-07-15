import numpy as np
import cv2 as cv
filename = 'C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/SIFT/Images/Boston_Mac.jpg'
img = cv.imread(filename)
#corners_mac = np.zeros(np.shape(gray))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print(gray.shape)
corners_mac = np.zeros((gray.shape[0],gray.shape[1],1))
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
corners_mac[dst>0.01*dst.max()]=255
cv.imshow('Corner Detections',corners_mac)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()