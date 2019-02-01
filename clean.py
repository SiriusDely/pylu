import cv2
import numpy as np

img = cv2.imread("./input/viewp-1.php.jpeg")

alpha = 2.0
beta = -160

new = alpha * img + beta
new = np.clip(new, 0, 255).astype(np.uint8)

cv2.imwrite("./output/viewp-1.php.cleaned.jpeg", new)
