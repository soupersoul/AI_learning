import cv2
import numpy as np

img = cv2.imread("./xuanye.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
binary = cv2.Canny(blur,100,255)
#ret, binary = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)
blured = cv2.medianBlur(binary, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
eroded = cv2.erode(blured, kernel)
contours, hierarchy= cv2.findContours(eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow("img", eroded)
#print(len(hierarchy), len(contours))
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y+h), (0, 0, 255), 1)
cv2.imshow("img", img)
cv2.waitKey(0)
