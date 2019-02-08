# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")

args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB
image = cv2.imread(args["image"])
# cv2.imshow("image", image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize(img, height=800):
    """ resize image to given height """
    ratio = height / img.shape[0]
    return cv2.resize(img, (int(ratio * img.shape[1]), height))

# resize and convert to grayscale
img = resize(image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img", img)

# bilateral filter preserve edges
img = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("img 2", img)

# create black and white image based on adaptive threshold
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
cv2.imshow("img 3", img)

# median filter clears small details
img = cv2.medianBlur(img, 11)
cv2.imshow("img 4", img)

# add black border in case that page is touching and image border
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
cv2.imshow("img 5", img)

edges = cv2.Canny(img, 200, 500)
cv2.imshow("edges", edges)

# getting contours
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# finding contour of biggest rectangle
# otherwise return corners of original image
# don't forget our 5px border
height = edges.shape[0]
width = edges.shape[1]
MAX_CONTOUR_AREA = (width - 10) * (height - 10)

# page fill at least half of image, then saving max area found
maxAreaFound = MAX_CONTOUR_AREA * 0.5

# saving page contour
pageContour = np.array([[5, 5], [5, height - 5], [width - 5, height - 5], [width - 5, 5]])

# loop through all the contours
for contour in contours:
    # simplify contour
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    # page has 4 corners and it is convex
    # page are must be bigger than maxAreaFound
    if (len(approx) == 4) and cv2.isContourConvex(approx) and maxAreaFound < cv2.contourArea(approx) < MAX_CONTOUR_AREA:
        maxAreaFound = cv2.contourArea(approx)
        pageContour = approx

# result in pageContour (numpy array of 4 points)
def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    
    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset
    
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt


# Sort and offset corners
pageContour = fourCornersSort(pageContour[:, 0])
pageContour = contourOffset(pageContour, (-5, -5))

# Recalculate to original scale - start Points
sPoints = pageContour.dot(image.shape[0] / 800)
  
# Using Euclidean distance
# Calculate maximum height (maximal length of vertical edges) and width
height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
             np.linalg.norm(sPoints[2] - sPoints[3]))
width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
             np.linalg.norm(sPoints[3] - sPoints[0]))

# Create target points
tPoints = np.array([[0, 0],
                    [0, height],
                    [width, height],
                    [width, 0]], np.float32)

# getPerspectiveTransform() needs float32
if sPoints.dtype != np.float32:
    sPoints = sPoints.astype(np.float32)

# Wraping perspective
M = cv2.getPerspectiveTransform(sPoints, tPoints) 
newImage = cv2.warpPerspective(image, M, (int(width), int(height)))

cv2.imshow("folder/resultImage.jpg", cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))

# Saving the result. Yay! (don't forget to convert colors bact to BGR)
# cv2.imwrite("folder/resultImage.jpg", cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))

cv2.waitKey(0)
cv2.destroyAllWindows()
