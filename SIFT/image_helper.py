import cv2

def make_quadrants(img):
    quadrants = []
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    l2 = img[:, :width_cutoff]
    l1 = img[:, width_cutoff:]
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    quadrants.append(l2)

    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    quadrants.append(l1)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r4 = img[:, :width_cutoff]
    r3 = img[:, width_cutoff:]
    # finish vertical devide image
    #rotate image to 90 COUNTERCLOCKWISE
    r4 = cv2.rotate(r4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    quadrants.append(r4)
    #rotate image to 90 COUNTERCLOCKWISE
    r3 = cv2.rotate(r3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #save
    quadrants.append(r3)
    return quadrants