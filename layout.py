# import packages
import numpy as np
import cv2

# load images
# image1 = cv2.imread('input/journal1.jpg')
image1 = cv2.imread('input/viewp-1.php.jpeg')

# outputs
output1_letter = image1.copy()

# grayscales
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# clean images using otsu emthod with the inversed binarized images
ret1, th1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# process letters boxing
def process_letter(thresh, output):
    # assign kernel size
    kernel = np.ones((2, 1), np.uint8) # vertical
    # use closing morph operation then erode to narrow the image
    temp_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    # temp_img = cv2.erode(thresh, kernel, iterations=2)
    letter_img = cv2.erode(temp_img, kernel, iterations=1)

    # find contours
    (contours, _) = cv2.findContours(letter_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop in all contour areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output, (x-1, y-5), (x+w, y+h), (0, 255, 0), 1)

    return output

# processing and writing the output
output1_letter = process_letter(th1, output1_letter)
cv2.imwrite('output/letter/output1_letter.jpg', output1_letter)

# show images
cv2.imshow('BW', output1_letter)
# wait for key 0
cv2.waitKey(0)
