# import necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the image from disk
image = cv2.imread(args["image"])

# convert image to grayscale and flip the back-fore-ground color
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

# threshold the image, foreground to 255 background to 0
retval, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("image", threshold)

# grab the (x, y) coordinates of all pixel values that are greater than zero,
# then use these coordinates to compute a rotated bounding box that contains
# all coordinates
coordinates = np.column_stack(np.where((threshold > 0)))
angle = cv2.minAreaRect(coordinates)[-1]

# the cv2.minAreaRect function returns values in the range of [-90, 0];
# as the rectangle rotates clockwise the returned angle trends to 0 --
# in this special case, we need to add 90 degrees to the angle.
if angle < -45:
    angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make it positive
else:
    angle = -angle

# rotate the image to deskew
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
cv2.imshow("rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
