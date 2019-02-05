# import necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing")

args = vars(ap.parse_args())

# load the image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess
if args["preprocess"] == "thresh":
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# check to see if median blurring should be done
elif args["preprocess"] == "blur":
    gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temp file so that we can apply OCR
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as PIL/Pillow image, apply OCR, and then delete the temp file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
