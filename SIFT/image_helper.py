import cv2


def make_quadrants(img):
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
    # rotate image to 90 COUNTERCLOCKWISE
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotate image to 90 COUNTERCLOCKWISE
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r4 = img[:, :width_cutoff]
    r3 = img[:, width_cutoff:]

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


# takes a macconnell image and pixel coords and retuns the length in inches from the bottom left corner of the image
# that the pixel is
def pixel2inches(mac, x, y):
    y = mac.shape[0] - y
    return x / 600, y / 600
