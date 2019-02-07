# import necessary packages
import argparse

import cv2
import imutils
import numpy as np

from imutils import paths

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
        help="path to the images directory")
args = vars(ap.parse_args())

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# loop over the input image paths
for imagePath in paths.list_images(args["images"]):
    # load the image, resize it, and convert it to grayscale
    image = cv2.imread(imagePath)
    image = imutils.resize(image, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth the image using 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    cv2.imshow("blackhat", blackhat)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    retval, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("thresh", thresh)

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform
    # a series of erosions to break apart connected components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    cv2.imshow("thresh 2", thresh)

    # during the thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    cv2.imshow("thresh 3", thresh)

    # find the contours in the threshold image and sort them by size
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # loop over the contours
    for contour in contours:
        # compute the bounding box of the contour and use it to
        # compute aspect ratio and coverage ratio of the bounding box's
        # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(contour)
        aspectRatio = w / float(h)
        coverageWidth = w / float(gray.shape[1])

        # check to see if the aspect ratio and coverage width are within
        # acceptable criteria
        if aspectRatio > 5 and coverageWidth > 0.75:
            # pad the bounding box since we applied erosions and now
            # need to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            # extract the ROI from the image and draw a bounding box
            # surrounding the MRZ
            roi = image[y:y + h, x:x + w].copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            break

    # show the output images
    cv2.imshow("image", image)
    cv2.imshow("roi", roi)


cv2.waitKey(0)
cv2.destroyAllWindows()
