# import necessary packages
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
ap.add_argument("-r", "--reference", required=True, help="path to the reference OCR-A image")
args = vars(ap.parse_args())

# define the dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
        }

# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, suct that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on *black*
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
_, ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow("ref", ref)

# find the contours in the OCR-A image (i.e., the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refContours = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refContours = imutils.grab_contours(refContours)
refContours = contours.sort_contours(refContours, method="left-to-right")[0]
digits = {}

# loop over the OCR-A reference contours
for (index, contour) in enumerate(refContours):
    # compute the bounding box for the digit, extract it, and resize
    # it to a fixed size
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # update the digits dictionary, mapping the digit name to the ROI
    digits[index] = roi

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

cv2.imshow("tophat", tophat)

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range(0, 255)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

cv2.imshow("gradX", gradX)

# apply a closing operation using the rectangular kernel to help
# close gaps in between credit card number digits, then apply
# Otsu's thresholding method to binarize the image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2.imshow("gradX 2", gradX)
_, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)

# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
gradX = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow("gradX 3", gradX)

# find the contours in the thresholded image, then initialize the
# list of digit locations
threshContours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
threshContours = imutils.grab_contours(threshContours)
locations = []

# loop over the contours
for (index, contour) in enumerate(threshContours):
    # compute the bounding box of the contour, then use the
    # bounding box coordinates to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(contour)
    aspectRatio = w / float(h)

    # since credit cards used a fixed size fonts with 4 groups
    # of 4 digits, we can prune potential contours based on the
    # aspect ratio
    if aspectRatio > 2.5 and aspectRatio < 4.0:
        # contours can further be pruned on minimum/maximum width and
        # height
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # append the bounding box region of the digits group
            # to our locations list
            locations.append((x, y, w, h))

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locations = sorted(locations, key=lambda x:x[0])
output = []

# loop over the 4 groupings of 4 digits
for (idx, (gX, gY, gW, gH)) in enumerate(locations):
    # initialize the list of group digits
    groupOutput = []

    # extract the group ROI of 4 digits from the grayscale image,
    # then apply thresholding to segment the digits from the background
    # of the credit card
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv2.imshow("group", group)
    _, group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("group 2", group)

    # detect the contours of each individual digit in the group,
    # then sort the digit contours from left to right
    digitContours = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitContours = imutils.grab_contours(digitContours)
    digitContours = contours.sort_contours(digitContours, method="left-to-right")[0]

    # loop over the digit contours
    for contour in digitContours:
        # compute the bounding box of the individual digit, extract
        # the digit, and resize it to have the same fixed size as
        # the reference OCR-A images
        (x, y, w, h) = cv2.boundingRect(contour)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # initialize a list of template matching scores
        scores = []
        
        # loop over the reference digit name and digit ROI
        for (digit, digitRoi) in digits.items():
            # apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, digitRoi, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # the classification for the digit ROI will be the reference
        # digit name with the *largest* template matching score
        groupOutput.append(str(np.argmax(scores)))

    # draw the digit classifications around the group
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5),
            (0, 0, 255), 2)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # update the output digits list
    output.extend(groupOutput)

# display the output credit card information to the screen
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))

cv2.imshow("image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
